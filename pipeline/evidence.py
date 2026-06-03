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

  Layer 1 — Keyword pre-selection (guaranteed)
    Scan each candidate's text for exact occurrences of the key terms that
    were extracted from the CP value during collation.  Any passage that
    contains a key term verbatim is guaranteed to be included — it cannot
    be dropped by the LLM.  This ensures that passages containing "SONIA",
    "GBP", or specific figures are never missed.

  Layer 2 — LLM selection (additive)
    The LLM reads all candidates and selects any additional passages it
    judges relevant that the keyword pass may have missed (e.g. passages
    that rephrase a concept without using the exact term).

The final result is the union of both layers, keyword-matched passages
listed first (highest priority), then any LLM-only additions.

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

    selected_sections    : the subset of CollatedResult.evidence kept as relevant.
                           Keyword-matched sections appear first.
    keyword_matched_count: how many sections were kept by the keyword pass.
    llm_added_count      : how many additional sections were added by the LLM.
    dropped_count        : how many candidates were filtered out entirely.
    filtered_context     : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list       # list[SectionEvidence]
    keyword_matched_count: int
    llm_added_count: int
    dropped_count: int
    filtered_context: str


# ── Keyword pre-selection ─────────────────────────────────────────────────────

# Terms shorter than this are too generic to be reliable keyword signals
# (e.g. "or", "at", "of").  Raise to exclude short financial abbreviations
# like "pa" if they cause false positives; lower to catch e.g. "USD".
_MIN_TERM_CHARS = 3


def _keyword_preselect(
    candidates: list[SectionEvidence],
    terms: list[str],
) -> tuple[list[int], list[int]]:
    """
    Scan each candidate section (heading + excerpt) for exact, case-insensitive
    occurrences of any key term.

    Returns
    -------
    (matched_indices, unmatched_indices)
        matched_indices   — positions in `candidates` that contain at least one term
        unmatched_indices — positions that contain none of the terms
    """
    # Build the set of meaningful search tokens from the extracted terms.
    # Each multi-word term is also split into its constituent words so that
    # e.g. "Final Maturity Date" matches a passage containing just "Maturity Date".
    token_set: set[str] = set()
    for term in terms:
        term = term.strip()
        if len(term) >= _MIN_TERM_CHARS:
            token_set.add(term.lower())
        for word in term.split():
            if len(word) >= _MIN_TERM_CHARS:
                token_set.add(word.lower())

    matched: list[int] = []
    unmatched: list[int] = []

    for i, sec in enumerate(candidates):
        searchable = (sec.heading + " " + sec.excerpt).lower()
        if any(token in searchable for token in token_set):
            matched.append(i)
        else:
            unmatched.append(i)

    return matched, unmatched


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

    Layer 1 — Keyword pre-selection (guaranteed inclusion)
        Any passage whose text contains a key term from the CP value is
        automatically included.  This is a hard guarantee — the LLM cannot
        drop these.  Handles exact financial terms like "SONIA", "GBP",
        specific figures, and defined terms.

    Layer 2 — LLM selection (additive)
        The LLM reads all candidates and identifies any further relevant
        passages that the keyword pass may have missed.  Its selections are
        added to the keyword-matched set; it cannot remove them.

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
            keyword_matched_count=0,
            llm_added_count=0,
            dropped_count=0,
            filtered_context="(No FA passages were retrieved for this CP field.)",
        )

    candidates: list[SectionEvidence] = collated.evidence
    n = len(candidates)

    # ── Layer 1: Keyword pre-selection ────────────────────────────────────────
    # Use the key terms that run_collate already extracted — these are the
    # precise legal/financial terms from the CP value.
    terms = collated.terms_extracted

    keyword_indices, remaining_indices = _keyword_preselect(candidates, terms)

    keyword_matched = [candidates[i] for i in keyword_indices]
    keyword_set = set(keyword_indices)

    print(f"         → Keyword pre-selected {len(keyword_matched)}/{n} passage(s): "
          f"{[candidates[i].heading for i in keyword_indices]}")

    # ── Layer 2: LLM selection over ALL candidates ────────────────────────────
    # Show the LLM the full candidate list so it has the complete picture.
    # Its role is to find passages the keyword pass missed — its selections
    # are added on top of the keyword matches, never replacing them.
    llm_added: list[SectionEvidence] = []

    if remaining_indices:   # only call LLM if there are non-keyword candidates to judge
        candidates_text = _format_candidates(candidates, max_chars_per_section)

        llm = ChatOpenAI(
            model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are reviewing candidate passages retrieved from a facility agreement.\n\n"
                "Your task: given a CP form value, identify passages from the list below "
                "that are most closely and directly relevant to it.\n\n"
                "A passage is relevant if it:\n"
                "  - Directly defines, states, or governs what the CP value describes\n"
                "  - Contains the specific rate, amount, date, condition, or mechanism "
                "referenced in the CP value\n\n"
                "A passage is NOT relevant if it:\n"
                "  - Only mentions a related term in passing or in a different context\n"
                "  - Covers a different clause or topic that happens to share a keyword\n"
                "  - Is general boilerplate with no specific bearing on the CP value\n\n"
                "Note: some passages have already been selected by a keyword match and "
                "will be included regardless. Focus on identifying any ADDITIONAL passages "
                "you judge relevant that may have been missed.\n\n"
                "Respond with ONLY the index numbers of relevant passages, "
                "comma-separated, most relevant first.\n"
                "Example: 2,0,4\n"
                "If none of the remaining passages are relevant, respond with: NONE"
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

        if raw.upper() != "NONE":
            seen_llm: set[int] = set()
            for token in re.findall(r'\d+', raw):
                idx = int(token)
                if 0 <= idx < n and idx not in keyword_set and idx not in seen_llm:
                    llm_added.append(candidates[idx])
                    seen_llm.add(idx)

        print(f"         → LLM added {len(llm_added)} further passage(s): "
              f"{[s.heading for s in llm_added]}")

    # ── Merge: keyword-matched first, then LLM additions ─────────────────────
    selected = keyword_matched + llm_added
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
        keyword_matched_count=len(keyword_matched),
        llm_added_count=len(llm_added),
        dropped_count=dropped,
        filtered_context=filtered_context,
    )
