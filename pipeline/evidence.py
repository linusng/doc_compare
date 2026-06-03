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
overlap with the CP value.  Not all of them are directly relevant.

`extract_evidence` collapses that list in two layers:

  Layer 1 — LLM filter (primary)
    The LLM reads all candidates and selects only those passages that are
    most directly and specifically relevant to the CP value.  It acts as a
    strict filter: aim for the smallest set that fully covers the CP value.

  Layer 2 — Specific-token safety net (additive)
    After the LLM has made its selection, scan the passages it dropped for
    highly specific tokens extracted directly from the CP value: all-caps
    benchmark names (SONIA, LIBOR, GBP), exact numbers, and percentages.
    Any dropped passage that contains such a token is added back — this
    ensures the LLM cannot accidentally discard a passage that contains a
    verbatim financial figure or benchmark name.

The final result is the union of both layers, LLM selections listed first,
then any safety-net additions.

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

    selected_sections  : the subset of CollatedResult.evidence kept as relevant.
                         LLM selections appear first, safety-net additions after.
    llm_selected_count : how many sections the LLM chose.
    safety_added_count : how many dropped sections were rescued by the safety net.
    dropped_count      : how many candidates were filtered out entirely.
    filtered_context   : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list       # list[SectionEvidence]
    llm_selected_count: int
    safety_added_count: int
    dropped_count: int
    filtered_context: str


# ── Specific-token extraction (safety net) ────────────────────────────────────

# Minimum length for a token to be used in the safety net
_MIN_TOKEN_CHARS = 3

# Matches all-uppercase abbreviations / benchmark names: SONIA, GBP, LIBOR, SOFR
_CAPS_TOKEN_RE = re.compile(r'\b[A-Z]{3,}\b')

# Matches numbers and percentages: 0.8%, 50,000,000, 1.5
_NUMBER_RE = re.compile(r'\b\d[\d,\.]*%?\b')


def _extract_specific_tokens(cp_value: str) -> set[str]:
    """
    Pull highly specific tokens out of the CP value for use as a safety net.

    Captures:
      - All-caps abbreviations / benchmarks  (SONIA, LIBOR, GBP, EUR, SOFR)
      - Numbers and percentages              (0.8%, 50,000,000, 3)

    These are the tokens we cannot afford to miss — if a candidate contains
    one verbatim, it stays even if the LLM filtered it out.
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
    max_chars_per_section: int = 1000,
) -> str:
    """
    Render the candidate passages as a numbered list for the LLM.
    Each entry shows the section heading, pages, and a truncated excerpt.
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
    max_chars_per_section: int = 1000,
) -> ExtractedEvidence:
    """
    Collapse the collated FA candidate passages down to those most directly
    relevant to the CP value, using a two-layer approach:

    Layer 1 — LLM filter (primary)
        The LLM reviews all candidates and selects the smallest subset that
        fully covers the CP value.  It is instructed to be strict — only
        passages that directly define, state, or govern the specific term,
        rate, amount, or date mentioned in the CP value should be kept.

    Layer 2 — Specific-token safety net (additive)
        Any passage that the LLM dropped but that contains a highly specific
        token from the CP value (all-caps benchmarks like SONIA, currency
        codes, exact numbers/percentages) is added back.  This prevents the
        LLM from accidentally discarding passages with verbatim financial
        figures or benchmark names.

    Parameters
    ----------
    cp_field              : CP form field name, e.g. "Margin".
    cp_value              : CP form value, e.g. "GBP: SONIA + 0.8% p.a.".
    collated              : CollatedResult from run_collate().
    max_chars_per_section : Max characters shown per candidate in the LLM prompt.

    Returns
    -------
    ExtractedEvidence
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

    candidates: list[SectionEvidence] = collated.evidence
    n = len(candidates)

    # ── Layer 1: LLM filter ───────────────────────────────────────────────────
    # The LLM sees all candidates and selects only the most directly relevant.
    # It is the primary decision-maker — we want strict filtering here.
    candidates_text = _format_candidates(candidates, max_chars_per_section)

    llm = ChatOpenAI(
        model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are filtering candidate passages retrieved from a facility agreement (FA).\n\n"
            "Your task: given a CP field name and its value, select ONLY the passages that "
            "are most directly and specifically relevant to it.\n\n"
            "KEEP a passage if it:\n"
            "  - Directly defines, states, or governs the specific term, rate, amount, "
            "date, or mechanism described in the CP value\n"
            "  - Contains the specific rate, benchmark, figure, or condition referenced "
            "(e.g. if the CP value mentions SONIA, keep passages that actually set or "
            "describe the SONIA-based margin or rate)\n\n"
            "DISCARD a passage if it:\n"
            "  - Only mentions a related term in passing or in a different context\n"
            "  - Covers a different clause that happens to share a keyword\n"
            "  - Is general boilerplate, definitions, or administrative text with no "
            "specific bearing on this CP value\n"
            "  - Discusses the concept generally without the specific values, rates, "
            "or dates stated in the CP value\n\n"
            "Be STRICT. Aim for the smallest set of passages that fully covers the "
            "CP value. If only 1 or 2 passages are truly relevant, return only those.\n\n"
            "Respond with ONLY the index numbers of the passages you are KEEPING, "
            "comma-separated, most relevant first.\n"
            "Example: 2,0\n"
            "If none of the passages are relevant, respond with: NONE"
        )),
        ("human", (
            "CP FIELD : {cp_field}\n"
            "CP VALUE : {cp_value}\n\n"
            "CANDIDATE PASSAGES ({n} total):\n\n"
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

    print(f"         → LLM selected {len(llm_selected)}/{n} passage(s): "
          f"{[candidates[i].heading for i in llm_selected_indices]}")

    # ── Layer 2: Specific-token safety net ────────────────────────────────────
    # Among the passages the LLM dropped, rescue any that contain a highly
    # specific token from the CP value (all-caps benchmarks, numbers, percentages).
    specific_tokens = _extract_specific_tokens(cp_value)
    print(f"         → Safety-net tokens: {specific_tokens}")

    safety_added: list[SectionEvidence] = []
    if specific_tokens:
        for i, sec in enumerate(candidates):
            if i in llm_selected_set:
                continue   # already kept
            searchable = (sec.heading + " " + sec.excerpt).lower()
            if any(tok in searchable for tok in specific_tokens):
                safety_added.append(sec)
                print(f"         → Safety net rescued: [{i}] {sec.heading}")

    # ── Merge: LLM selections first, then safety-net rescues ─────────────────
    selected = llm_selected + safety_added
    dropped = n - len(selected)

    # Fallback: if both layers returned nothing, keep all candidates.
    if not selected:
        print("         → Both layers returned nothing — keeping all candidates as fallback")
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
