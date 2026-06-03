"""
Evidence extraction — the stage between collation and comparison.

Pipeline position
-----------------
  run_collate()        → CollatedResult   broad retrieval (many candidate passages)
  extract_evidence()   → ExtractedEvidence  ← this module
  analyze_deviation()  → DeviationAnalysis  comparison against CP value

What this stage does
--------------------
`run_collate` returns every FA passage that is semantically related to the
extracted terms from the CP value.  That net is deliberately wide — recall
over precision — so not every returned passage is directly relevant.

`extract_evidence` narrows the net:

1. Decomposes the CP value into its discrete items / aspects.
   e.g.  "GBP: SONIA + 0.8% p.a."
         → aspect 1: currency (GBP)
         → aspect 2: benchmark rate (SONIA)
         → aspect 3: spread (0.8% per annum)

2. For each aspect, identifies which FA section(s) from the collated
   evidence actually address it, and extracts the precise sentence(s) that
   are directly relevant.

3. Collapses everything into a clean `filtered_context` string that the
   downstream comparison agent can use without wading through noise.

Python API
----------
    from pipeline.collate import run_collate
    from pipeline.evidence import extract_evidence

    collated = run_collate(index, "Margin: GBP SONIA + 0.8% p.a.")
    evidence = extract_evidence("Margin", "GBP: SONIA + 0.8% p.a.", collated)

    print(evidence.filtered_context)   # feed this to analyze_deviation
    for item in evidence.items:
        print(item.cp_aspect, "→", item.fa_section, item.fa_pages)
"""

import json
import re

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .models import CollatedResult, SectionEvidence


# ── Output models ─────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """
    One aspect of the CP value, mapped to the FA passage that addresses it.
    """
    cp_aspect: str      # the specific item from the CP value, e.g. "benchmark rate"
    fa_section: str     # section/clause heading in the FA, e.g. "8.1 Interest"
    fa_pages: list      # page numbers where this was found
    fa_passage: str     # the precise sentence(s) from the FA relevant to this aspect
    relevance_note: str # brief explanation of why this passage addresses the aspect


class ExtractedEvidence(BaseModel):
    """
    Filtered evidence for one CP field: one EvidenceItem per distinct aspect
    of the CP value.
    """
    cp_field: str
    cp_value: str
    items: list           # list[EvidenceItem]
    unmatched_aspects: list  # CP aspects for which no FA evidence was found
    filtered_context: str    # collapsed, clean context ready for analyze_deviation


# ── Prompt helpers ────────────────────────────────────────────────────────────

def _format_evidence_for_prompt(evidence: list[SectionEvidence], max_chars_per_section: int = 1200) -> str:
    """
    Render the collated SectionEvidence list as a numbered block for the
    LLM prompt.  Each section gets an index so the LLM can reference it
    by number in its response.

    Excerpts are truncated to max_chars_per_section to stay within context
    limits while preserving the opening text (where definitions and key
    terms typically appear).
    """
    parts = []
    for i, sec in enumerate(evidence):
        excerpt = sec.excerpt.strip()
        if len(excerpt) > max_chars_per_section:
            excerpt = excerpt[:max_chars_per_section] + "\n[… truncated]"

        parts.append(
            f"[Section {i}]  {sec.heading}  (pages {sec.pages})\n"
            f"Matched terms: {', '.join(sec.matched_terms)}\n"
            f"{'-' * 50}\n"
            f"{excerpt}"
        )
    return "\n\n".join(parts)


def _render_filtered_context(items: list[EvidenceItem]) -> str:
    """
    Render a clean, structured context string from the extracted evidence
    items.  This is what gets passed to the downstream comparison agent.
    """
    if not items:
        return "(No directly relevant FA passages found.)"

    parts = []
    for item in items:
        parts.append(
            f"CP ASPECT  : {item.cp_aspect}\n"
            f"FA SECTION : {item.fa_section}  (pages {item.fa_pages})\n"
            f"FA PASSAGE : {item.fa_passage}\n"
            f"NOTE       : {item.relevance_note}"
        )
    return "\n\n" + ("\n" + "─" * 60 + "\n").join(parts)


# ── Core extraction function ──────────────────────────────────────────────────

def extract_evidence(
    cp_field: str,
    cp_value: str,
    collated: CollatedResult,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    max_chars_per_section: int = 1200,
) -> ExtractedEvidence:
    """
    Filter and structure the collated FA passages down to only the evidence
    that is directly relevant to the CP field/value.

    Parameters
    ----------
    cp_field   : Name of the CP form field, e.g. "Margin".
    cp_value   : Value stated in the CP form, e.g. "GBP: SONIA + 0.8% p.a.".
    collated   : The CollatedResult returned by run_collate().
    max_chars_per_section : Characters shown per FA section in the prompt.
                 Increase if sections are being truncated and important text
                 is being missed.  Default 1200 is safe for most models.

    Returns
    -------
    ExtractedEvidence
        items             — one EvidenceItem per CP aspect that was found in the FA
        unmatched_aspects — aspects the LLM could not find evidence for
        filtered_context  — formatted string ready for the comparison agent
    """
    if not collated.evidence:
        return ExtractedEvidence(
            cp_field=cp_field,
            cp_value=cp_value,
            items=[],
            unmatched_aspects=["(collation returned no candidate passages)"],
            filtered_context="(No FA passages were retrieved for this CP field.)",
        )

    fa_sections_text = _format_evidence_for_prompt(
        collated.evidence, max_chars_per_section=max_chars_per_section,
    )

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst specialising in facility agreements.\n\n"
            "You will be given:\n"
            "  CP FIELD   — the name of a term from a conditions precedent (CP) form\n"
            "  CP VALUE   — the value or description stated in the CP form for that term\n"
            "  FA SECTIONS — candidate passages retrieved from the facility agreement (FA),\n"
            "                numbered [Section 0], [Section 1], etc.\n\n"

            "STEP 1 — Decompose the CP VALUE into its distinct aspects / items.\n"
            "  e.g. 'GBP: SONIA + 0.8% p.a.' → currency, benchmark rate, spread\n"
            "  e.g. 'Two extensions of 1 year each' → number of extensions, duration each\n"
            "  e.g. 'USD 50,000,000' → currency, amount\n\n"

            "STEP 2 — For each aspect, scan the FA SECTIONS and find the passage that\n"
            "  most directly states or defines that aspect.\n"
            "  Extract the precise sentence(s) from the FA (do not paraphrase).\n"
            "  If no passage addresses an aspect, list it under 'unmatched_aspects'.\n\n"

            "Rules:\n"
            "  - Only include passages that DIRECTLY address the CP value aspect.\n"
            "  - Do not include general background or unrelated clauses.\n"
            "  - If multiple sections say the same thing, pick the most authoritative\n"
            "    (definitions section > operative clause > schedule).\n"
            "  - Quotes in fa_passage must be verbatim from the FA SECTIONS text.\n\n"

            "Return ONLY valid JSON in this exact shape (no markdown, no commentary):\n"
            "{\n"
            '  "items": [\n'
            "    {\n"
            '      "cp_aspect": "benchmark rate",\n'
            '      "fa_section": "8.1 Interest",\n'
            '      "fa_pages": [12],\n'
            '      "fa_passage": "Interest shall accrue at SONIA plus the Margin...",\n'
            '      "relevance_note": "Confirms SONIA as the benchmark for GBP loans"\n'
            "    }\n"
            "  ],\n"
            '  "unmatched_aspects": ["aspect that had no FA evidence"]\n'
            "}"
        )),
        ("human", (
            "CP FIELD  : {cp_field}\n"
            "CP VALUE  : {cp_value}\n\n"
            "FA SECTIONS:\n{fa_sections}"
        )),
    ])

    raw = (prompt | llm).invoke({
        "cp_field":   cp_field,
        "cp_value":   cp_value,
        "fa_sections": fa_sections_text,
    }).content.strip()

    # ── Parse JSON response ───────────────────────────────────────────────────
    items: list[EvidenceItem] = []
    unmatched: list[str] = []

    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())

            for entry in parsed.get("items", []):
                try:
                    items.append(EvidenceItem(
                        cp_aspect=str(entry.get("cp_aspect", "")).strip(),
                        fa_section=str(entry.get("fa_section", "")).strip(),
                        fa_pages=entry.get("fa_pages", []),
                        fa_passage=str(entry.get("fa_passage", "")).strip(),
                        relevance_note=str(entry.get("relevance_note", "")).strip(),
                    ))
                except Exception:
                    continue  # skip malformed individual entries

            unmatched = [
                str(a).strip()
                for a in parsed.get("unmatched_aspects", [])
                if str(a).strip()
            ]

        except (json.JSONDecodeError, TypeError):
            # JSON parse failed entirely — fall back to returning the raw
            # combined_context so the comparison agent still has something to work with
            items = []
            unmatched = ["[Evidence extraction parse error — using raw collated context]"]

    filtered_context = _render_filtered_context(items)

    # Fallback: if the LLM returned nothing useful, use the raw combined_context
    # so the downstream comparison agent is never left with an empty input.
    if not items:
        filtered_context = (
            "[Evidence extraction returned no structured items. "
            "Raw collated context follows:]\n\n"
            + collated.combined_context[:4000]
        )

    return ExtractedEvidence(
        cp_field=cp_field,
        cp_value=cp_value,
        items=items,
        unmatched_aspects=unmatched,
        filtered_context=filtered_context,
    )
