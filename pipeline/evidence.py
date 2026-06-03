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

`extract_evidence` collapses that list in two steps:

  Step 1 — Part-chunk deduplication
    Merge candidates that share the same base heading (stripping "(part N)"
    suffixes) into single entries so each FA section appears once.

  Step 2 — Strict token-score filter
    Extract every meaningful token from both the CP field name and the CP
    value (all-caps abbreviations, numbers, percentages, and significant
    words).  Score each candidate by the number of distinct CP tokens it
    contains.  Keep only candidates whose score meets the threshold:

      • score ≥ 2  when the CP value has ≥ 2 distinct tokens
      • score ≥ 1  when the CP value has only 1 distinct token

    If nothing clears the threshold, the single highest-scoring candidate
    is kept as a fallback so the downstream comparison never runs blind.

This is intentionally strict — passages that only share a generic keyword
with the CP value (e.g. the word "interest" appearing in a boilerplate
clause) will score 0 or 1 and be dropped.

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

from pydantic import BaseModel

from .models import CollatedResult, SectionEvidence


# ── Output model ──────────────────────────────────────────────────────────────

class ExtractedEvidence(BaseModel):
    """
    Filtered evidence for one CP field.

    selected_sections : candidates that passed the token-score filter.
    kept_count        : number of sections kept.
    dropped_count     : number of sections dropped.
    filtered_context  : formatted string ready for the comparison agent.
    """
    cp_field: str
    cp_value: str
    selected_sections: list   # list[SectionEvidence]
    kept_count: int
    dropped_count: int
    filtered_context: str


# ── Token extraction ──────────────────────────────────────────────────────────

# Noise words excluded from field-name token extraction
_STOP_WORDS = {
    "the", "and", "or", "of", "in", "to", "a", "an", "for",
    "from", "with", "by", "as", "at", "on", "is", "its",
}

# Regex patterns for high-specificity tokens from the CP value
_CAPS_RE   = re.compile(r'\b[A-Z]{2,}\b')          # SONIA, GBP, LIBOR, USD
_NUMBER_RE = re.compile(r'\b\d[\d,\.]*%?\b')        # 0.8%, 50,000,000, 3
_PART_SUFFIX_RE = re.compile(r'\s*\(part\s*\d+\)\s*$', re.IGNORECASE)


def _cp_value_tokens(cp_value: str) -> set[str]:
    """
    Extract every meaningful token from the CP value.

    Captures:
      - All-caps abbreviations / benchmarks  : SONIA, GBP, LIBOR, EURIBOR
      - Numbers and percentages              : 0.8%, 50000000, 3.5
      - Significant mixed-case words (≥4 ch) : "Margin", "Maturity", "Extension"

    These tokens are the direct subject matter of the CP value — a relevant
    FA passage must contain at least some of them.
    """
    tokens: set[str] = set()

    # All-caps abbreviations
    for m in _CAPS_RE.finditer(cp_value):
        tokens.add(m.group().lower())

    # Numbers / percentages
    for m in _NUMBER_RE.finditer(cp_value):
        tok = m.group().lower()
        if len(tok) >= 2:
            tokens.add(tok)

    # Significant words (≥4 chars, not stop words)
    for word in re.findall(r'[A-Za-z]{4,}', cp_value):
        w = word.lower()
        if w not in _STOP_WORDS:
            tokens.add(w)

    return tokens


def _cp_field_tokens(cp_field: str) -> set[str]:
    """
    Extract significant words from the CP field name (≥4 chars, not stop words).
    Used only as a secondary signal — not counted toward the score threshold.
    """
    tokens: set[str] = set()
    for word in re.findall(r'[A-Za-z]{4,}', cp_field):
        w = word.lower()
        if w not in _STOP_WORDS:
            tokens.add(w)
    return tokens


# ── Part-chunk deduplication ──────────────────────────────────────────────────

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


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score(sec: SectionEvidence, value_tokens: set[str]) -> int:
    """
    Count how many distinct CP-value tokens appear in the section text
    (heading + excerpt, case-insensitive).
    """
    searchable = (sec.heading + " " + sec.excerpt).lower()
    return sum(1 for tok in value_tokens if tok in searchable)


# ── Formatting ────────────────────────────────────────────────────────────────

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
    **_kwargs,   # absorbs llm_model / base_url / api_key — no longer needed
) -> ExtractedEvidence:
    """
    Filter collated FA passages down to those that directly address the CP value.

    Step 1 — Deduplicate part-chunks (merge "(part N)" fragments per section).
    Step 2 — Score each section by the number of CP-value tokens it contains.
              Keep only sections that meet the minimum score threshold.
    """
    if not collated.evidence:
        return ExtractedEvidence(
            cp_field=cp_field,
            cp_value=cp_value,
            selected_sections=[],
            kept_count=0,
            dropped_count=0,
            filtered_context="(No FA passages were retrieved for this CP field.)",
        )

    # ── Step 1: Merge part-chunks ─────────────────────────────────────────────
    candidates = _merge_part_chunks(collated.evidence)
    n_raw = len(collated.evidence)
    n = len(candidates)
    print(f"         → Deduplicated {n_raw} chunk(s) → {n} section(s)")

    # ── Step 2: Token-score filter ────────────────────────────────────────────
    value_tokens = _cp_value_tokens(cp_value)
    field_tokens = _cp_field_tokens(cp_field)
    all_tokens   = value_tokens | field_tokens

    print(f"         → CP value tokens : {sorted(value_tokens)}")
    print(f"         → CP field tokens : {sorted(field_tokens)}")

    # Threshold: must match at least 2 value tokens when ≥2 exist, else 1.
    threshold = min(2, len(value_tokens)) if value_tokens else 1

    scored: list[tuple[int, SectionEvidence]] = []
    for sec in candidates:
        s = _score(sec, value_tokens)
        scored.append((s, sec))
        print(f"         → [{s:2d}] {sec.heading}")

    selected = [sec for s, sec in scored if s >= threshold]

    # Fallback: if nothing passes, keep the single highest-scoring section.
    if not selected:
        best_score, best_sec = max(scored, key=lambda x: x[0])
        print(f"         → Nothing passed threshold ({threshold}) — "
              f"keeping best scorer: {best_sec.heading} (score={best_score})")
        selected = [best_sec]

    dropped = n - len(selected)
    print(f"         → Kept {len(selected)}/{n} section(s) "
          f"(threshold={threshold}, dropped={dropped})")

    return ExtractedEvidence(
        cp_field=cp_field,
        cp_value=cp_value,
        selected_sections=selected,
        kept_count=len(selected),
        dropped_count=dropped,
        filtered_context=_render_filtered_context(selected),
    )
