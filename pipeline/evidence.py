"""
Evidence extraction — the stage between collation and comparison.

Pipeline position
-----------------
  run_collate()       → CollatedResult      broad retrieval, many candidate passages
  extract_evidence()  → ExtractedEvidence   ← this module, filters to relevant ones
  analyze_deviation() → DeviationAnalysis   comparison against CP value

What this stage does
--------------------
`run_collate` casts a wide net — it retrieves every FA passage whose terms
overlap with the CP value.  Not all of them are directly relevant, and the
same section may appear as several numbered "part" chunks.

`extract_evidence` collapses that list in three steps:

  Step 1 — Part-chunk deduplication
    Merge candidate sections that share the same base heading (stripping
    " (part N)" suffixes) so the LLM sees whole sections, not fragments.

  Step 2 — LLM filter (primary)
    The LLM reads the deduplicated list and selects the SMALLEST set of
    sections that fully covers the CP value.  Typically 1–3 sections for
    a well-defined CP field.

  Step 3 — Specific-token safety net (additive)
    After the LLM has made its selection, scan the passages it dropped for
    highly specific tokens extracted directly from the CP value: all-caps
    benchmark names (SONIA, GBP, LIBOR), exact numbers, and percentages.
    Any dropped passage that contains such a token is added back.

Python API
----------
    from pipeline.collate import run_collate
    from pipeline.evidence import extract_evidence

    collated = run_collate(index, "Margin: GBP SONIA + 0.8% p.a.")
    evidence = extract_evidence("Margin", "GBP: SONIA + 0.8% p.a.", collated)

    print(evidence.filtered_context)     # feed directly to analyze_deviation
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

    selected_sections  : the subset of CollatedResult.evidence kept as relevant
                         (after deduplication and merging of part-chunks).
                         LLM selections appear first, safety-net additions after.
    llm_selected_count : how many (deduplicated) sections the LLM chose.
    safety_added_count : how many dropped sections were rescued by the safety net.
    dropped_count      : how many (deduplicated) candidates were filtered out.
    filtered_context   : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list       # list[SectionEvidence]
    llm_selected_count: int
    safety_added_count: int
    dropped_count: int
    filtered_context: str


# ── Step 1: Part-chunk deduplication ─────────────────────────────────────────

_PART_SUFFIX_RE = re.compile(r'\s*\(part\s*\d+\)\s*$', re.IGNORECASE)


def _merge_part_chunks(sections: list[SectionEvidence]) -> list[SectionEvidence]:
    """
    Merge SectionEvidence entries that come from the same base section
    (i.e. their headings differ only by a ' (part N)' suffix).

    Parts are merged in order: pages are unioned, excerpts are concatenated
    with a separator, and matched_terms are unioned.  The result is a shorter
    list of complete sections rather than multiple fragments.
    """
    merged: dict[str, dict] = {}   # base_heading → accumulated fields
    order: list[str] = []           # insertion order

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
            # union pages (preserve order, no duplicates)
            for p in sec.pages:
                if p not in acc["pages"]:
                    acc["pages"].append(p)
            # union matched terms
            for t in sec.matched_terms:
                if t not in acc["matched_terms"]:
                    acc["matched_terms"].append(t)
            # concatenate excerpt
            acc["excerpt"] += "\n\n" + sec.excerpt

    return [
        SectionEvidence(**merged[base])
        for base in order
    ]


# ── Step 3: Specific-token safety net ─────────────────────────────────────────

_MIN_TOKEN_CHARS = 3

# All-caps abbreviations / benchmarks: SONIA, GBP, LIBOR, SOFR, EURIBOR
_CAPS_TOKEN_RE = re.compile(r'\b[A-Z]{3,}\b')

# Numbers and percentages: 0.8%, 50,000,000, 3
_NUMBER_RE = re.compile(r'\b\d[\d,\.]*%?\b')


def _extract_specific_tokens(cp_value: str) -> set[str]:
    """
    Pull highly specific tokens out of the CP value for the safety net:
      - All-caps abbreviations / benchmarks  (SONIA, GBP, LIBOR)
      - Numbers and percentages              (0.8%, 50,000,000)

    These are tokens we cannot afford to miss — if a dropped candidate
    contains one verbatim, it is added back.
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
    """
    Render the deduplicated candidate passages as a numbered list for the LLM.
    """
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
    """
    Format the selected sections as a clean context block for the
    downstream comparison agent.
    """
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
    Collapse the collated FA candidate passages down to those most directly
    relevant to the CP value.

    Step 1 — Deduplication
        Merge part-chunks (same base heading) so the LLM sees whole sections.

    Step 2 — LLM filter
        The LLM selects the smallest set of sections that fully covers the CP
        value.  Instructed to be strict: typically 1–3 sections.

    Step 3 — Safety net
        Any section dropped by the LLM but containing a highly specific token
        from the CP value (all-caps benchmark, exact number/%) is added back.
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

    # ── Step 1: Deduplicate part-chunks ───────────────────────────────────────
    candidates: list[SectionEvidence] = _merge_part_chunks(collated.evidence)
    n_raw = len(collated.evidence)
    n = len(candidates)

    print(f"         → Deduplicated {n_raw} chunk(s) → {n} section(s)")

    # ── Step 2: LLM filter ────────────────────────────────────────────────────
    candidates_text = _format_candidates(candidates, max_chars_per_section)

    llm = ChatOpenAI(
        model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are filtering candidate passages from a facility agreement (FA) "
            "down to only those relevant to a specific CP form field.\n\n"
            "KEEP a passage ONLY if it:\n"
            "  - Directly states, defines, or governs the specific term, rate, "
            "amount, date, or condition described in the CP value\n"
            "  - Contains the specific benchmark, figure, or mechanism referenced "
            "(e.g. if CP value mentions SONIA + 0.8%, keep passages that actually "
            "set that rate — not passages that merely mention interest rates in passing)\n\n"
            "DISCARD a passage if it:\n"
            "  - Only uses a related keyword in a different context\n"
            "  - Is a general definition, administrative provision, or boilerplate\n"
            "  - Covers a related but distinct clause (e.g. a different interest rate, "
            "a different tranche, an unrelated condition)\n"
            "  - Repeats the same substance already covered by a passage you are keeping\n\n"
            "TARGET: return the FEWEST passages possible — typically 1 to 3 for a "
            "well-defined CP field.  Only return more if the CP value has genuinely "
            "distinct aspects each requiring a separate passage.\n\n"
            "Respond with ONLY the index numbers of passages you are KEEPING, "
            "comma-separated, most relevant first.\n"
            "Example: 2,0\n"
            "If none are relevant, respond: NONE"
        )),
        ("human", (
            "CP FIELD : {cp_field}\n"
            "CP VALUE : {cp_value}\n\n"
            "CANDIDATE PASSAGES ({n} total — select the fewest that fully cover the CP value):\n\n"
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

    # ── Step 3: Safety net ────────────────────────────────────────────────────
    specific_tokens = _extract_specific_tokens(cp_value)
    print(f"         → Safety-net tokens: {specific_tokens}")

    safety_added: list[SectionEvidence] = []
    if specific_tokens:
        for i, sec in enumerate(candidates):
            if i in llm_selected_set:
                continue
            searchable = (sec.heading + " " + sec.excerpt).lower()
            if any(tok in searchable for tok in specific_tokens):
                safety_added.append(sec)
                print(f"         → Safety net rescued: [{i}] {sec.heading}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    selected = llm_selected + safety_added
    dropped = n - len(selected)

    if not selected:
        print("         → Both layers empty — keeping all deduplicated sections as fallback")
        selected = candidates
        dropped = 0

    filtered_context = _render_filtered_context(selected)

    return ExtractedEvidence(
        cp_field=cp_field,
        cp_value=cp_value,
        selected_sections=selected,
        llm_selected_count=len(llm_selected),
        safety_added_count=len(safety_added),
        dropped_count=dropped,
        filtered_context=filtered_context,
    )
