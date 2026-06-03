"""
Evidence extraction — the stage between collation and comparison.

Pipeline position
-----------------
  run_collate()       → CollatedResult      broad retrieval, many candidate passages
  extract_evidence()  → ExtractedEvidence   ← this module, filters to relevant ones
  analyze_deviation() → DeviationAnalysis   comparison against CP value

Why an LLM filter (not keyword matching)
----------------------------------------
The facility agreement may be written in a *different language* than the CP
form.  A CP value of "Margin: SONIA + 0.8% p.a." might need to match an FA
clause written in Chinese, French, or German.  Pure keyword/token matching
cannot bridge that gap — only an LLM that understands both languages can tell
that a passage *means* the same thing.  So the LLM is the primary filter.

What this stage does
--------------------
`run_collate` casts a wide net.  `extract_evidence` collapses it in two steps:

  Step 1 — Part-chunk deduplication
    Merge candidates that share the same base heading (stripping "(part N)"
    suffixes) so each FA section appears once — fewer, whole sections for the
    LLM to judge, instead of many fragments.

  Step 2 — Strict multilingual LLM filter
    The LLM reads the deduplicated candidates and keeps ONLY those whose
    content directly states/defines/governs the specific thing in the CP
    value — regardless of the language the passage is written in.  It is told
    to be ruthless: when in doubt, drop.  Target is the fewest passages
    (typically 1–3) that fully cover the CP value.

  Safety net — language-neutral tokens
    Some tokens mean the same in every language: exact numbers (0.8%,
    50,000,000), benchmark codes (SONIA, LIBOR, SOFR), currency codes
    (GBP, USD, EUR).  If a passage the LLM dropped contains one of these
    verbatim from the CP value, it is added back — these are too specific
    to risk losing to an LLM misjudgement.

Python API
----------
    from pipeline.collate import run_collate
    from pipeline.evidence import extract_evidence

    collated = run_collate(index, "Margin: GBP SONIA + 0.8% p.a.")
    evidence = extract_evidence("Margin", "GBP: SONIA + 0.8% p.a.", collated)

    print(evidence.filtered_context)
    for sec in evidence.selected_sections:
        print(sec.heading, sec.pages)
"""

import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .models import CollatedResult, SectionEvidence


# ── Output model ──────────────────────────────────────────────────────────────

class ExtractedEvidence(BaseModel):
    """
    Filtered evidence for one CP field.

    selected_sections  : sections kept (LLM selections first, safety net after).
    llm_selected_count : how many sections the LLM kept.
    safety_added_count : how many dropped sections the safety net rescued.
    dropped_count      : how many candidates were filtered out entirely.
    filtered_context   : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list   # list[SectionEvidence]
    llm_selected_count: int
    safety_added_count: int
    dropped_count: int
    filtered_context: str


# ── Part-chunk deduplication ──────────────────────────────────────────────────

_PART_SUFFIX_RE = re.compile(r'\s*\(part\s*\d+\)\s*$', re.IGNORECASE)


def _merge_part_chunks(sections: list[SectionEvidence]) -> list[SectionEvidence]:
    """
    Merge SectionEvidence entries that share the same base heading
    (differing only by a " (part N)" suffix).  Pages and matched_terms
    are unioned; excerpts are concatenated.
    """
    merged: dict[str, dict] = {}
    order: list[str] = []

    for sec in sections:
        base = _PART_SUFFIX_RE.sub("", sec.heading).strip()

        if base not in merged:
            merged[base] = {
                "heading":       base,
                "pages":         list(sec.pages),
                "matched_terms": list(sec.matched_terms),
                "excerpt":       sec.excerpt,
            }
            order.append(base)
        else:
            acc = merged[base]
            for p in sec.pages:
                if p not in acc["pages"]:
                    acc["pages"].append(p)
            for t in sec.matched_terms:
                if t not in acc["matched_terms"]:
                    acc["matched_terms"].append(t)
            acc["excerpt"] += "\n\n" + sec.excerpt

    return [SectionEvidence(**merged[base]) for base in order]


# ── Language-neutral safety-net tokens ────────────────────────────────────────

_MIN_TOKEN_CHARS = 3

# Tokens that read identically in any language: SONIA, GBP, LIBOR, SOFR, USD …
_CAPS_TOKEN_RE = re.compile(r'\b[A-Z]{3,}\b')
# Exact numbers / percentages: 0.8%, 50,000,000, 3
_NUMBER_RE = re.compile(r'\b\d[\d,\.]*%?\b')


def _language_neutral_tokens(cp_value: str) -> set[str]:
    """
    Extract tokens from the CP value that mean the same in every language —
    benchmark/currency codes and exact figures.  These survive translation,
    so they make a reliable cross-lingual safety net.
    """
    tokens: set[str] = set()
    for m in _CAPS_TOKEN_RE.finditer(cp_value):
        tok = m.group().lower()
        if len(tok) >= _MIN_TOKEN_CHARS:
            tokens.add(tok)
    for m in _NUMBER_RE.finditer(cp_value):
        tok = m.group().lower()
        if len(tok) >= _MIN_TOKEN_CHARS:
            tokens.add(tok)
    return tokens


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _format_candidates(
    candidates: list[SectionEvidence],
    max_chars_per_section: int = 1500,
) -> str:
    parts = []
    for i, sec in enumerate(candidates):
        excerpt = sec.excerpt.strip()
        if len(excerpt) > max_chars_per_section:
            excerpt = excerpt[:max_chars_per_section] + "\n[… truncated]"
        parts.append(
            f"[{i}] {sec.heading}  (pages {sec.pages})\n"
            f"{'─' * 50}\n"
            f"{excerpt}"
        )
    return "\n\n".join(parts)


def _render_filtered_context(sections: list[SectionEvidence]) -> str:
    if not sections:
        return "(No directly relevant FA passages found.)"
    parts = []
    for sec in sections:
        parts.append(
            f"SECTION : {sec.heading}  (pages {sec.pages})\n"
            f"{'─' * 50}\n"
            f"{sec.excerpt.strip()}"
        )
    return "\n\n".join(parts)


# ── Core function ─────────────────────────────────────────────────────────────

def extract_evidence(
    cp_field: str,
    cp_value: str,
    collated: CollatedResult,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    max_chars_per_section: int = 1500,
) -> ExtractedEvidence:
    """
    Filter collated FA passages down to those that directly address the CP value,
    working across languages.

    Step 1 — Deduplicate part-chunks.
    Step 2 — Strict multilingual LLM filter (primary).
    Safety net — re-add dropped passages containing language-neutral tokens
                 (benchmark/currency codes, exact figures) from the CP value.
    """
    if not collated.evidence:
        return ExtractedEvidence(
            cp_field=cp_field,
            cp_value=cp_value,
            selected_sections=[],
            llm_selected_count=0,
            safety_added_count=0,
            dropped_count=0,
            filtered_context="(No FA passages were retrieved for this CP field.)",
        )

    # ── Step 1: Merge part-chunks ─────────────────────────────────────────────
    candidates = _merge_part_chunks(collated.evidence)
    n_raw = len(collated.evidence)
    n = len(candidates)
    print(f"         → Deduplicated {n_raw} chunk(s) → {n} section(s)")

    # ── Step 2: Strict multilingual LLM filter ────────────────────────────────
    candidates_text = _format_candidates(candidates, max_chars_per_section)

    llm = ChatOpenAI(
        model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You filter candidate passages from a facility agreement (FA) down to "
            "only those relevant to one specific CP form field.\n\n"
            "IMPORTANT — LANGUAGE: The FA passages may be written in a DIFFERENT "
            "language than the CP field/value (e.g. the CP value is in English but "
            "a passage is in Chinese, French, German, etc.). Judge relevance by "
            "MEANING, not by matching words. A passage in another language that "
            "expresses the same term, rate, amount, date, or condition as the CP "
            "value IS relevant and must be kept.\n\n"
            "KEEP a passage ONLY if its meaning:\n"
            "  - Directly states, defines, or governs the specific term, rate, "
            "amount, date, or condition in the CP value\n"
            "  - Contains the specific benchmark, figure, or mechanism referenced "
            "(e.g. for 'SONIA + 0.8%', keep the clause that actually sets that "
            "margin — in ANY language — not a clause that merely mentions interest)\n\n"
            "DROP a passage if it:\n"
            "  - Only touches a related topic in a different context\n"
            "  - Is a general definition, administrative provision, or boilerplate\n"
            "  - Covers a related but DISTINCT clause (a different rate, tranche, "
            "or condition)\n"
            "  - Repeats substance already covered by a passage you are keeping\n\n"
            "BE RUTHLESS. When in doubt, DROP. Return the FEWEST passages that fully "
            "cover the CP value — typically 1 to 3. Only return more if the CP value "
            "has genuinely distinct aspects each needing a separate passage.\n\n"
            "Respond with ONLY the index numbers you are KEEPING, comma-separated, "
            "most relevant first. Example: 2,0\n"
            "If none are relevant, respond: NONE"
        )),
        ("human", (
            "CP FIELD : {cp_field}\n"
            "CP VALUE : {cp_value}\n\n"
            "CANDIDATE PASSAGES ({n} total — keep the fewest that fully cover the CP value):\n\n"
            "{candidates}"
        )),
    ])

    raw = (prompt | llm).invoke({
        "cp_field":   cp_field,
        "cp_value":   cp_value,
        "n":          n,
        "candidates": candidates_text,
    }).content.strip()

    llm_selected_indices: list[int] = []
    if raw.upper() != "NONE":
        seen: set[int] = set()
        for token in re.findall(r'\d+', raw):
            idx = int(token)
            if 0 <= idx < n and idx not in seen:
                llm_selected_indices.append(idx)
                seen.add(idx)

    llm_selected_set = set(llm_selected_indices)
    llm_selected = [candidates[i] for i in llm_selected_indices]

    print(f"         → LLM kept {len(llm_selected)}/{n}: "
          f"{[candidates[i].heading for i in llm_selected_indices]}")

    # ── Safety net: language-neutral tokens ───────────────────────────────────
    neutral_tokens = _language_neutral_tokens(cp_value)
    print(f"         → Language-neutral safety tokens: {neutral_tokens}")

    safety_added: list[SectionEvidence] = []
    if neutral_tokens:
        for i, sec in enumerate(candidates):
            if i in llm_selected_set:
                continue
            searchable = (sec.heading + " " + sec.excerpt).lower()
            if any(tok in searchable for tok in neutral_tokens):
                safety_added.append(sec)
                print(f"         → Safety net rescued: [{i}] {sec.heading}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    selected = llm_selected + safety_added
    dropped = n - len(selected)

    if not selected:
        print("         → LLM kept nothing and no neutral tokens matched — "
              "keeping all deduplicated sections as fallback")
        selected = candidates
        dropped = 0

    return ExtractedEvidence(
        cp_field=cp_field,
        cp_value=cp_value,
        selected_sections=selected,
        llm_selected_count=len(llm_selected),
        safety_added_count=len(safety_added),
        dropped_count=dropped,
        filtered_context=_render_filtered_context(selected),
    )
