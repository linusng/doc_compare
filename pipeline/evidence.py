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

`extract_evidence` collapses that list.  Given the CP value and the candidate
passages, the LLM selects only those passages that most closely and directly
address what the CP value is describing.  Everything else is dropped.

The result is a short, clean list of FA passages ready for the comparison agent.

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

    selected_sections : the subset of CollatedResult.evidence that the LLM
                        judged directly relevant to the CP value.
    filtered_context  : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list    # list[SectionEvidence] — kept passages
    dropped_count: int         # how many candidates were filtered out
    filtered_context: str      # clean context for analyze_deviation


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _format_candidates(
    evidence: list[SectionEvidence],
    max_chars_per_section: int = 1000,
) -> str:
    """
    Render the candidate passages as a numbered list for the LLM.
    Each entry shows the section heading, pages, and a truncated excerpt.
    """
    parts = []
    for i, sec in enumerate(evidence):
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
    Given the CP value and the candidate FA passages from run_collate,
    select only those passages that are most directly relevant.

    The LLM reads each candidate and picks the indices of the ones that
    closely and directly address what the CP value describes.  All others
    are dropped.

    Parameters
    ----------
    cp_field              : CP form field name, e.g. "Margin".
    cp_value              : CP form value, e.g. "GBP: SONIA + 0.8% p.a.".
    collated              : CollatedResult from run_collate().
    max_chars_per_section : Max characters shown per candidate in the prompt.
                            Increase if relevant text is being cut off.

    Returns
    -------
    ExtractedEvidence with selected_sections and filtered_context.
    """
    if not collated.evidence:
        return ExtractedEvidence(
            cp_field=cp_field,
            cp_value=cp_value,
            selected_sections=[],
            dropped_count=0,
            filtered_context="(No FA passages were retrieved for this CP field.)",
        )

    candidates: list[SectionEvidence] = collated.evidence
    n = len(candidates)

    candidates_text = _format_candidates(
        candidates, max_chars_per_section=max_chars_per_section,
    )

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are reviewing candidate passages retrieved from a facility agreement.\n\n"
            "Your task: given a CP form value, select the passages from the list below "
            "that are most closely and directly relevant to it.\n\n"
            "A passage is relevant if it:\n"
            "  - Directly defines, states, or governs what the CP value describes\n"
            "  - Contains the specific rate, amount, date, condition, or mechanism "
            "referenced in the CP value\n\n"
            "A passage is NOT relevant if it:\n"
            "  - Only mentions a related term in passing\n"
            "  - Covers a different clause or topic that happens to share a keyword\n"
            "  - Is general boilerplate with no specific bearing on the CP value\n\n"
            "Respond with ONLY the index numbers of the relevant passages, "
            "comma-separated, most relevant first.\n"
            "Example: 2,0,4\n"
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

    # ── Parse the response ────────────────────────────────────────────────────
    selected: list[SectionEvidence] = []

    if raw.upper() != "NONE":
        seen: set[int] = set()
        for token in re.findall(r'\d+', raw):
            idx = int(token)
            if 0 <= idx < n and idx not in seen:
                selected.append(candidates[idx])
                seen.add(idx)

    dropped = n - len(selected)
    filtered_context = _render_filtered_context(selected)

    # Fallback: if the LLM selected nothing or gave an unparseable response,
    # keep all candidates so the comparison agent is never left empty-handed.
    if not selected:
        filtered_context = (
            "[No passages were selected — returning all candidates as fallback]\n\n"
            + _render_filtered_context(candidates)
        )
        selected = candidates
        dropped = 0

    return ExtractedEvidence(
        cp_field=cp_field,
        cp_value=cp_value,
        selected_sections=selected,
        dropped_count=dropped,
        filtered_context=filtered_context,
    )
