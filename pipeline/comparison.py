"""
Deviation comparison — the Comparison Agent.

Main entry point: run_comparison(index, fields) -> pandas.DataFrame

Pipeline position
-----------------
    retrieval.retrieve()   → RetrievalResult   passages covering the CP field
    compare_field()        → FieldComparison   ← this module, the verdict
    run_comparison()       → pandas.DataFrame  the review sheet

Output columns
--------------
    S/N | CP Field | FA Legal Section | Deviation | Comments

What the agent actually does
----------------------------
The naive version of this — "here is a CP value, here are some passages, is
there a deviation?" — fails in three specific ways, and each stage below exists
to close one of them:

  1. It answers about the EASY part of the field and silently ignores the rest.
     A CP field is rarely atomic ("USD 250m term loan, quarterly repayment,
     Margin 2.25%, tested semi-annually"); a single free-text verdict tends to
     address the amount and never mention the testing frequency.
     → Fixed by comparing PER ITEM. retrieval.py has already decomposed the
       field into its constituent items (`RetrievalResult.aspects`); the agent
       must return a finding for EVERY item, and the Comments column is
       assembled from those findings, so nothing in the query can go unaddressed.

  2. It calls "absent" a match. If the FA simply does not mention a requirement,
     an LLM asked "is this consistent?" will usually say yes — nothing
     contradicted it.
     → Fixed by making NOT_ADDRESSED a first-class verdict, seeded
       deterministically from retrieval coverage: an item that retrieval could
       not cover in any round is reported as not addressed without asking the
       model, and (by default) counts as a deviation.

  3. It cites passages it did not use, so the evidence column becomes a dump.
     → Fixed by making the agent declare `sections` per finding. The FA Legal
       Section column contains ONLY passages tied to a finding — the passages
       the comparison actually turned on — not everything retrieval returned.

On top of that the module applies a deterministic NUMERIC GUARD: amounts,
percentages and ratios are extracted from the CP field and from the cited
passages and compared as data, not as prose. A model that reads "not less than
2.25 per cent." and writes "matches" is caught when the FA actually says 2.50.

Everything degrades: with no LLM reachable the retrieval evidence, the coverage
report and the numeric guard are still produced, and the row is marked for
manual review rather than silently asserting "No deviation".

Python API
----------
    from pipeline.ingestion_v3 import DocumentIndexV3
    from pipeline.comparison import run_comparison, compare_field

    index = DocumentIndexV3.from_pdf("facility_agreement.pdf")

    # one line item per CP field — dict, list or JSON file all work
    df = run_comparison(index, {
        "Facility Amount": "USD 250,000,000 term loan",
        "Margin":          "2.25% per annum over Term SOFR",
    })
    df.to_excel("cp_review.xlsx", index=False)

    # single field, with the full structured result
    result = compare_field(index, "Margin: 2.25% per annum over Term SOFR")
    print(result.comments)
    for f in result.findings:
        print(f.verdict, f.item_text, f.fa_position)

CLI
---
    python -m pipeline.comparison --pdf fa.pdf --cp cp_fields.json --out review.xlsx
"""

import argparse
import json
import re
import sys
import unicodedata

import pandas as pd
from pydantic import BaseModel, Field

from .retrieval import (
    RetrievalResult,
    RetrievedItem,
    Retriever,
    LLM,
    parse_json_array,
)


# ── Tunables ──────────────────────────────────────────────────────────────────

MIN_EVIDENCE_GRADE = 2       # retrieval grade a passage needs to reach the agent
MAX_EVIDENCE_SECTIONS = 12   # passages shown to the agent per field
EVIDENCE_SNIPPET_CHARS = 1400
MAX_COMMENT_CHARS = 2000     # cap on the assembled Comments cell
SECTION_SEPARATOR = "\n\n"   # between passages in the FA Legal Section cell

# Verdicts a finding can carry. NOT_ADDRESSED is deliberately distinct from
# DEVIATION: "the FA says something different" and "the FA is silent" are
# different review outcomes, even though both default to Deviation = Yes.
V_MATCH = "match"
V_DEVIATION = "deviation"
V_PARTIAL = "partial"
V_NOT_ADDRESSED = "not_addressed"
V_UNKNOWN = "unknown"          # not a verdict the agent may return — see below
_VERDICTS = {V_MATCH, V_DEVIATION, V_PARTIAL, V_NOT_ADDRESSED}

_VERDICT_LABEL = {
    V_MATCH: "matches",
    V_DEVIATION: "DEVIATION",
    V_PARTIAL: "PARTIAL",
    V_NOT_ADDRESSED: "NOT ADDRESSED IN FA",
    V_UNKNOWN: "NOT ASSESSED",
}

# Verdicts that make the row a deviation. NOT_ADDRESSED is included by default —
# a requirement the FA never picks up is the most commonly missed deviation of
# all — and can be switched off with treat_missing_as_deviation=False.
#
# V_UNKNOWN is deliberately NOT here. "The agent could not assess this" is not a
# finding of deviation, and quietly recording it as one poisons the sheet: every
# row goes Yes the moment the endpoint is down, and a reviewer learns to ignore
# the column. Unknown items set needs_review instead, so the row says so out loud
# while the deterministic numeric guard can still raise a Yes on its own evidence.
_DEVIATING = {V_DEVIATION, V_PARTIAL}


# ── Normalisation (kept local: this module owns its own text handling) ────────

_PUNCT_TABLE = str.maketrans({
    "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "",
})


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _flatten(text: str) -> str:
    """Collapse a passage to a single line — the FA Legal Section cell format."""
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


# ── Page labelling ────────────────────────────────────────────────────────────

def _page_label(pages: list, offset: int = 0) -> str:
    """
    '(Page 33)' / '(Pages 33-34)' / '(Pages 33, 36)'.

    `offset` is added to every page number, so a caller can convert the parser's
    0-based indices into the document's printed page numbers (DocumentIndexV3
    exposes `page_offset` for exactly this).
    """
    nums = []
    for p in pages or []:
        try:
            nums.append(int(p) + offset)
        except (TypeError, ValueError):
            continue
    nums = sorted(set(nums))
    if not nums:
        return ""
    if len(nums) == 1:
        return f"(Page {nums[0]})"
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"(Pages {nums[0]}-{nums[-1]})"
    return "(Pages " + ", ".join(str(n) for n in nums) + ")"


def format_fa_section(
    item: RetrievedItem,
    page_offset: int = 0,
    include_heading: bool = False,
) -> str:
    """
    Render one passage for the FA Legal Section cell:

        <content, newlines collapsed to spaces> (Page N)

    include_heading=True prepends '<heading> — ' , which is useful when the sheet
    is read by a human but is off by default to keep the specified format exact.
    """
    body = _flatten(item.content) or _flatten(item.heading)
    label = _page_label(item.pages, page_offset)
    head = f"{_flatten(item.heading)} — " if include_heading and item.heading else ""
    return f"{head}{body} {label}".strip()


# ── Numeric guard (deterministic value comparison) ────────────────────────────
#
# Prose comparison is where LLMs are strong; number comparison is where they are
# weak and where the cost of an error is highest. Every figure is therefore also
# checked as data.

_AMOUNT_RE = re.compile(
    r"(?:(?:usd|eur|gbp|sgd|jpy|chf|aud|hkd|cny|rmb)\s?|[$£€¥]\s?)"
    r"([\d,]+(?:\.\d+)?)\s?(million|billion|bn|m\b|k\b)?",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent\.?|percent)", re.IGNORECASE)
_RATIO_RE = re.compile(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)")

_MULTIPLIER = {"million": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "k": 1e3}


class ValueFlag(BaseModel):
    """One numeric discrepancy between the CP field and the cited FA passages."""
    kind: str                      # "amount" | "percentage" | "ratio"
    cp_value: str                  # as written in the CP field
    fa_values: list[str] = Field(default_factory=list)   # same-kind values found
    found_in_fa: bool = False      # the CP figure appears verbatim in the evidence

    def describe(self) -> str:
        if self.found_in_fa:
            return ""
        if self.fa_values:
            return (f"CP {self.kind} {self.cp_value} not found in the cited FA text, "
                    f"which states {', '.join(self.fa_values[:3])}")
        return f"CP {self.kind} {self.cp_value} not found in the cited FA text"


def _extract_values(text: str) -> dict[str, list[tuple[str, float | None]]]:
    """
    Pull comparable figures out of text, grouped by kind.

    Returns {kind: [(as_written, numeric_value_or_None), ...]}. Amounts are
    scaled ("USD 250 million" → 250000000.0) so a CP written in millions and an
    FA written in full can be compared at all.
    """
    out: dict[str, list[tuple[str, float | None]]] = {
        "amount": [], "percentage": [], "ratio": []}
    t = text or ""

    for m in _AMOUNT_RE.finditer(t):
        raw = " ".join(m.group(0).split())
        try:
            value = float(m.group(1).replace(",", ""))
        except (TypeError, ValueError):
            value = None
        unit = (m.group(2) or "").lower().strip()
        if value is not None and unit in _MULTIPLIER:
            value *= _MULTIPLIER[unit]
        out["amount"].append((raw, value))

    for m in _PERCENT_RE.finditer(t):
        try:
            out["percentage"].append((" ".join(m.group(0).split()), float(m.group(1))))
        except (TypeError, ValueError):
            continue

    for m in _RATIO_RE.finditer(t):
        try:
            out["ratio"].append((" ".join(m.group(0).split()),
                                 float(m.group(1)) / (float(m.group(2)) or 1.0)))
        except (TypeError, ValueError, ZeroDivisionError):
            continue
    return out


def check_values(cp_text: str, evidence_text: str) -> list[ValueFlag]:
    """
    Compare every figure in the CP field against the cited FA text.

    A flag is raised when a CP figure has no numeric equal in the evidence.
    Comparison is on VALUE, not on spelling, so "USD 250,000,000" and
    "USD 250 million" agree, while 2.25% and 2.50% do not.
    """
    cp_vals = _extract_values(cp_text)
    fa_vals = _extract_values(evidence_text)
    flags: list[ValueFlag] = []

    for kind, entries in cp_vals.items():
        fa_entries = fa_vals.get(kind, [])
        fa_numbers = [v for _, v in fa_entries if v is not None]
        for raw, value in entries:
            if value is None:
                continue
            found = any(abs(value - other) <= max(1e-9, abs(value) * 1e-6)
                        for other in fa_numbers)
            if found:
                continue
            flags.append(ValueFlag(
                kind=kind, cp_value=raw, found_in_fa=False,
                fa_values=[w for w, _ in fa_entries][:5],
            ))
    return flags


# ── Result models ─────────────────────────────────────────────────────────────

class EvidenceRef(BaseModel):
    """One FA passage the comparison actually relied on."""
    chunk_id: int
    heading: str
    pages: list = Field(default_factory=list)
    content: str
    grade: int = 0
    rationale: str | None = None      # the reranker's one-line justification
    formatted: str = ""               # the FA Legal Section rendering

    def __str__(self) -> str:
        return f"[{self.heading}] {_page_label(self.pages)}"


class AspectFinding(BaseModel):
    """
    The verdict on ONE item of the CP field.

    This is the unit that makes the Comments column complete: the field's items
    come from retrieval's decomposition, and every one of them ends up here with
    a verdict, so no part of the query can be silently skipped.
    """
    item_id: int
    item_text: str
    verdict: str = V_NOT_ADDRESSED
    fa_position: str = ""             # what the FA says on this item
    note: str = ""                    # why it does / does not deviate
    chunk_ids: list[int] = Field(default_factory=list)

    @property
    def is_deviation(self) -> bool:
        return self.verdict in _DEVIATING

    def line(self) -> str:
        label = _VERDICT_LABEL.get(self.verdict, self.verdict)
        parts = [f"• {self.item_text.rstrip('.')} — {label}."]
        if self.fa_position:
            parts.append(f"FA: {self.fa_position.rstrip('.')}.")
        if self.note:
            note = self.note.rstrip(".")
            parts.append(note[:1].upper() + note[1:] + ".")
        return " ".join(parts)


class FieldComparison(BaseModel):
    """
    The full result for one CP field — one row of the review sheet, plus the
    working that produced it.
    """
    cp_field: str
    deviation: bool = False
    comments: str = ""
    summary: str = ""
    sections: list[EvidenceRef] = Field(default_factory=list)
    findings: list[AspectFinding] = Field(default_factory=list)
    value_flags: list[ValueFlag] = Field(default_factory=list)
    coverage: float = 0.0
    llm_used: bool = False
    needs_review: bool = False        # agent unavailable / unparseable output
    stats: dict = Field(default_factory=dict)

    @property
    def deviation_label(self) -> str:
        return "Yes" if self.deviation else "No"

    @property
    def fa_legal_section(self) -> str:
        """The FA Legal Section cell: only the passages tied to a finding."""
        return SECTION_SEPARATOR.join(s.formatted for s in self.sections)

    def to_row(self) -> dict:
        return {
            "CP Field": self.cp_field,
            "FA Legal Section": self.fa_legal_section,
            "Deviation": self.deviation_label,
            "Comments": self.comments,
        }


class ComparisonReport(BaseModel):
    """Every field compared, with the DataFrame view."""
    results: list[FieldComparison] = Field(default_factory=list)

    def __iter__(self):
        return iter(self.results)

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, i):
        return self.results[i]

    @property
    def deviations(self) -> list[FieldComparison]:
        return [r for r in self.results if r.deviation]

    def to_dataframe(self) -> pd.DataFrame:
        """S/N | CP Field | FA Legal Section | Deviation | Comments."""
        df = pd.DataFrame([r.to_row() for r in self.results],
                          columns=["CP Field", "FA Legal Section", "Deviation", "Comments"])
        df.insert(0, "S/N", range(1, len(df) + 1))
        return df


# ── Evidence preparation ──────────────────────────────────────────────────────

def select_evidence(
    result: RetrievalResult,
    min_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_EVIDENCE_SECTIONS,
) -> list[RetrievedItem]:
    """
    Narrow retrieval's output to the passages worth showing the agent.

    Retrieval deliberately returns extras — weak matches for starved items,
    definition-graph supports, sibling parts — because recall is its job.
    Precision is this module's job: keep confidently-graded passages, then admit
    supports (definitions the clauses depend on) with the leftover budget.
    """
    primary = [it for it in result.items if not it.is_support and it.grade >= min_grade]
    primary.sort(key=lambda it: (-it.grade, -it.score))

    # Nothing clears the bar → show the weak matches (grade 1) so the agent can
    # rule them out itself, but never grade-0 noise: an unrelated clause in the
    # FA Legal Section column reads as evidence and is worse than an empty cell.
    if not primary:
        primary = sorted((it for it in result.items
                          if not it.is_support and it.grade >= 1),
                         key=lambda it: (-it.grade, -it.score))[:max_sections]

    chosen = primary[:max_sections]
    room = max_sections - len(chosen)
    if room > 0:
        chosen += [it for it in result.items if it.is_support][:room]
    return chosen


def build_evidence_block(items: list[RetrievedItem], page_offset: int = 0) -> str:
    """
    Render the candidate passages for the agent.

    Each passage carries its retrieval metadata — grade, the reranker's
    rationale, and which items of the CP field it was retrieved for. That
    supplementary signal is what lets the agent tell "this is the operative
    clause for item 2" from "this is a definition that happens to share
    vocabulary", instead of re-deriving relevance from the text alone.
    """
    blocks = []
    for it in items:
        meta = [f"pages {it.pages}"]
        if not it.is_support:
            meta.append(f"retrieval grade {it.grade}/3")
            if it.covers:
                meta.append(f"retrieved for item(s) {it.covers}")
        else:
            meta.append(f"supporting context via {it.via}")
        if it.rationale:
            meta.append(f"relevance: {it.rationale}")
        blocks.append(
            f"### SECTION {it.chunk_id}\n"
            f"HEADING: {it.heading}\n"
            f"META: {'; '.join(meta)}\n"
            f"TEXT: {it.content.strip()[:EVIDENCE_SNIPPET_CHARS]}"
        )
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


# ── The Comparison Agent ──────────────────────────────────────────────────────

_COMPARISON_SYSTEM = (
    "You are a legal document analyst comparing a CP (conditions precedent / term "
    "sheet) field against the signed facility agreement (FA).\n\n"
    "You are given:\n"
    "  CP FIELD  — what the CP form requires.\n"
    "  ITEMS     — the CP field broken into its individual requirements. You must "
    "return exactly one finding per item, using the item's id.\n"
    "  SECTIONS  — FA passages retrieved for this field, each with an id, its "
    "retrieval grade and why it was retrieved.\n\n"
    "For EACH item decide:\n"
    '  "match"         — the FA provides the same thing as the CP item.\n'
    '  "deviation"     — the FA provides something DIFFERENT (different figure, '
    "different mechanism, extra condition, more or less restrictive).\n"
    '  "partial"       — the FA covers the item only in part, or on different terms '
    "in some respect.\n"
    '  "not_addressed" — nothing in the SECTIONS addresses this item. Use this '
    "rather than guessing; never treat FA silence as agreement.\n\n"
    "Rules:\n"
    "- Judge ONLY from the SECTIONS text. Never assume the FA contains something "
    "you were not shown.\n"
    "- Compare figures, dates, periods and thresholds EXACTLY: 2.25% vs 2.50%, "
    "quarterly vs semi-annually, and USD vs EUR are deviations.\n"
    "- For each finding, list the section ids you relied on. Only list a section "
    "you actually used.\n"
    "- fa_position must state what the FA ACTUALLY SAYS on that item, concretely "
    "(quote the operative figure or wording), not 'the FA addresses this'.\n\n"
    "Return ONLY a JSON array, one object per item:\n"
    '  {"item": <id>, "verdict": "match"|"deviation"|"partial"|"not_addressed", '
    '"fa_position": "<what the FA says, max 30 words>", '
    '"note": "<why it does or does not deviate, max 25 words>", '
    '"sections": [<section id>, ...]}'
)


def run_agent(
    cp_field: str,
    result: RetrievalResult,
    evidence: list[RetrievedItem],
    llm: LLM,
) -> list[dict]:
    """
    Ask the agent for one finding per item of the CP field.

    Returns the raw parsed findings (validated downstream). An empty list means
    the agent was unavailable or produced nothing usable — the caller then falls
    back to the deterministic path rather than inventing a verdict.
    """
    if not llm.available or not result.aspects:
        return []

    items_block = "\n".join(f"  [{a.id}] {a.text}" for a in result.aspects)
    human = (
        f"CP FIELD:\n{cp_field}\n\n"
        f"ITEMS ({len(result.aspects)} — return one finding for each):\n{items_block}\n\n"
        f"SECTIONS:\n{build_evidence_block(evidence)}"
    )
    return [e for e in parse_json_array(llm.chat(_COMPARISON_SYSTEM, human))
            if isinstance(e, dict)]


def _normalise_findings(
    raw: list[dict],
    result: RetrievalResult,
    evidence_by_id: dict[int, RetrievedItem],
) -> list[AspectFinding]:
    """
    Turn the agent's output into one validated finding per item.

    Two guarantees are enforced here, both of them things the model gets wrong
    often enough to matter:
      • EVERY item gets a finding. Items the agent skipped are filled in from
        retrieval's own coverage — an item retrieval never covered is reported
        as not addressed, which is the honest answer.
      • Only REAL section ids survive. A cited id that was not in the evidence
        set is dropped, so the FA Legal Section column cannot contain a passage
        that was never shown to the agent.
    """
    by_item: dict[int, dict] = {}
    for entry in raw:
        try:
            item_id = int(entry.get("item"))
        except (TypeError, ValueError):
            continue
        by_item.setdefault(item_id, entry)

    findings: list[AspectFinding] = []
    for aspect in result.aspects:
        entry = by_item.get(aspect.id)
        if entry is None:
            # The agent skipped this item — fall back to retrieval coverage.
            # Retrieval found nothing → genuinely not addressed. Retrieval found
            # something the agent never ruled on → unknown, not a deviation.
            covered = aspect.covered
            findings.append(AspectFinding(
                item_id=aspect.id, item_text=aspect.text,
                verdict=V_UNKNOWN if covered else V_NOT_ADDRESSED,
                note=("passages were retrieved for this item but the agent returned "
                      "no finding — manual review"
                      if covered else "no FA passage found for this item"),
                chunk_ids=[c for c in aspect.evidence if c in evidence_by_id][:3],
            ))
            continue

        verdict = str(entry.get("verdict", "")).strip().lower()
        if verdict not in _VERDICTS:
            verdict = V_UNKNOWN if aspect.covered else V_NOT_ADDRESSED

        ids = []
        for sid in entry.get("sections") or []:
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            if sid in evidence_by_id and sid not in ids:
                ids.append(sid)

        # A "match" with no citation is unsupported — the agent is asserting the
        # FA says something without pointing at where. It becomes unassessed
        # (flagged for review), not a deviation: an uncited claim is missing
        # evidence, which is not the same as evidence of a difference.
        if verdict == V_MATCH and not ids:
            verdict = V_UNKNOWN
            entry = {**entry, "note": (str(entry.get("note", "") or "").strip()
                                       + " (match asserted without citing a section)").strip()}

        findings.append(AspectFinding(
            item_id=aspect.id, item_text=aspect.text, verdict=verdict,
            fa_position=str(entry.get("fa_position", "") or "").strip(),
            note=str(entry.get("note", "") or "").strip(),
            chunk_ids=ids,
        ))
    return findings


def _compose_comments(
    findings: list[AspectFinding],
    value_flags: list[ValueFlag],
    coverage: float,
    needs_review: bool,
    max_chars: int = MAX_COMMENT_CHARS,
) -> str:
    """
    Assemble the Comments cell so it addresses the WHOLE field.

    Deterministic assembly from the per-item findings (rather than asking the
    model for a free-text paragraph) is what guarantees coverage: the cell opens
    with the verdict count, then carries one line per item — what the FA says and
    why it does or does not deviate — followed by any numeric discrepancies.
    A model writing prose would drop the boring items; this cannot.
    """
    lines: list[str] = []

    deviating = [f for f in findings if f.verdict == V_DEVIATION]
    partial = [f for f in findings if f.verdict == V_PARTIAL]
    missing = [f for f in findings if f.verdict == V_NOT_ADDRESSED]
    unknown = [f for f in findings if f.verdict == V_UNKNOWN]
    matched = [f for f in findings if f.verdict == V_MATCH]

    head = []
    if deviating:
        head.append(f"{len(deviating)} deviation(s)")
    if partial:
        head.append(f"{len(partial)} partial")
    if missing:
        head.append(f"{len(missing)} not addressed in FA")
    if unknown:
        head.append(f"{len(unknown)} not assessed")
    if matched:
        head.append(f"{len(matched)} matching")
    if head:
        lines.append(f"{len(findings)} item(s) checked: " + ", ".join(head) + ".")

    # Deviations first — that is what a reviewer reads the cell for.
    for group in (deviating, partial, missing, unknown, matched):
        for f in group:
            lines.append(f.line())

    for flag in value_flags:
        note = flag.describe().rstrip(".")
        if note:
            lines.append(f"⚠ {note}.")

    if needs_review:
        lines.append("⚠ One or more items could not be assessed by the comparison "
                     "agent — manual review required.")
    elif coverage < 1.0:
        lines.append(f"Retrieval covered {coverage:.0%} of the field's items.")

    text = "\n".join(lines).strip()
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


# ── Per-field comparison ──────────────────────────────────────────────────────

def compare_field(
    index_or_retriever,
    cp_field: str,
    *,
    retrieval_result: RetrievalResult | None = None,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    use_llm: bool = True,
    min_evidence_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_EVIDENCE_SECTIONS,
    treat_missing_as_deviation: bool = True,
    numeric_guard: bool = True,
    page_offset: int = 0,
    include_heading: bool = False,
    verbose: bool = True,
    **retrieval_kwargs,
) -> FieldComparison:
    """
    Compare ONE CP field against the FA and return the structured verdict.

    Args:
        index_or_retriever: a DocumentIndex* or a pre-built retrieval.Retriever
                            (reuse the Retriever across fields — it holds the
                            BM25 index).
        cp_field:           the input query — free text, may cover several items.
        retrieval_result:   pre-computed retrieval output; retrieved here if omitted.
        min_evidence_grade: retrieval grade a passage needs to reach the agent.
        treat_missing_as_deviation:
                            an item no FA passage addresses counts as a deviation
                            (default True — FA silence is a review finding).
        numeric_guard:      cross-check amounts/percentages/ratios as numbers.
        page_offset:        added to page numbers in the FA Legal Section cell.
        include_heading:    prepend the clause heading to each rendered passage.
        **retrieval_kwargs: forwarded to Retriever.retrieve() (max_items, …).

    Returns:
        FieldComparison — .deviation, .comments, .sections, .findings, and
        .to_row() for the DataFrame.
    """
    retriever = (index_or_retriever if isinstance(index_or_retriever, Retriever)
                 else Retriever(index_or_retriever, llm_model=llm_model,
                                base_url=base_url, api_key=api_key,
                                use_llm=use_llm, verbose=verbose))

    if retrieval_result is None:
        retrieval_result = retriever.retrieve(cp_field, verbose=verbose,
                                              **retrieval_kwargs)

    evidence = select_evidence(retrieval_result, min_evidence_grade, max_sections)
    evidence_by_id = {it.chunk_id: it for it in evidence}

    llm = LLM(model=llm_model, base_url=base_url, api_key=api_key,
               enabled=use_llm, verbose=verbose)
    raw = run_agent(cp_field, retrieval_result, evidence, llm)
    findings = _normalise_findings(raw, retrieval_result, evidence_by_id)
    # The row needs a human whenever the agent produced nothing, or left any item
    # unassessed — those items carry no verdict and must not read as "No".
    needs_review = (not raw) or any(f.verdict == V_UNKNOWN for f in findings)

    # ── The FA Legal Section column: only passages a finding relied on ────────
    cited_ids: list[int] = []
    for f in findings:
        for cid in f.chunk_ids:
            if cid not in cited_ids:
                cited_ids.append(cid)
    # With no agent output there is nothing to cite from; fall back to the
    # confidently-graded passages so the row still shows its evidence. Weak
    # passages are NOT promoted here — if every item came back not addressed,
    # the honest cell is an empty one.
    if not cited_ids:
        cited_ids = [it.chunk_id for it in evidence
                     if not it.is_support and it.grade >= min_evidence_grade
                     ][:max_sections]

    sections = []
    for cid in cited_ids:
        it = evidence_by_id.get(cid)
        if it is None:
            continue
        sections.append(EvidenceRef(
            chunk_id=it.chunk_id, heading=it.heading, pages=it.pages,
            content=it.content, grade=it.grade, rationale=it.rationale,
            formatted=format_fa_section(it, page_offset, include_heading),
        ))

    # ── Numeric guard over exactly the cited text ─────────────────────────────
    value_flags: list[ValueFlag] = []
    if numeric_guard and sections:
        value_flags = check_values(cp_field, " ".join(s.content for s in sections))

    # ── Verdict ───────────────────────────────────────────────────────────────
    deviation = any(f.is_deviation for f in findings)
    if treat_missing_as_deviation:
        deviation = deviation or any(f.verdict == V_NOT_ADDRESSED for f in findings)
    # A CP figure that appears nowhere in the cited text, while the FA states a
    # different figure of the same kind, is a deviation regardless of the prose
    # verdict — this is the case models most often wave through.
    if numeric_guard and any(fl.fa_values and not fl.found_in_fa for fl in value_flags):
        deviation = True

    comments = _compose_comments(
        findings, value_flags, retrieval_result.coverage, needs_review)

    if verbose:
        print(f"      → Deviation: {'Yes' if deviation else 'No'} · "
              f"{len(sections)} section(s) cited · {len(findings)} item(s) assessed"
              + (f" · {len(value_flags)} numeric flag(s)" if value_flags else ""))

    return FieldComparison(
        cp_field=cp_field, deviation=deviation, comments=comments,
        summary=comments.split("\n")[0] if comments else "",
        sections=sections, findings=findings, value_flags=value_flags,
        coverage=retrieval_result.coverage, llm_used=llm.calls > 0,
        needs_review=needs_review,
        stats={
            "retrieved": len(retrieval_result.items),
            "evidence_shown": len(evidence),
            "sections_cited": len(sections),
            "items": len(findings),
            "llm_calls": llm.calls,
            "retrieval": retrieval_result.stats,
        },
    )


# ── Batch comparison ──────────────────────────────────────────────────────────

def _as_queries(fields) -> list[str]:
    """
    Accept the CP fields in any convenient shape and return one query per line
    item:  {"Margin": "2.25%"} → "Margin: 2.25%";  ["...", "..."] → as given;
    a path to a JSON file → loaded first.
    """
    if isinstance(fields, str):
        with open(fields, encoding="utf-8") as f:
            fields = json.load(f)
    if isinstance(fields, dict):
        return [f"{k}: {v}" for k, v in fields.items()]
    if isinstance(fields, (list, tuple)):
        out = []
        for entry in fields:
            if isinstance(entry, dict):          # [{"field": ..., "value": ...}]
                name = entry.get("field") or entry.get("name") or ""
                value = entry.get("value", "")
                out.append(f"{name}: {value}".strip(": ").strip())
            else:
                out.append(str(entry))
        return out
    raise TypeError(f"Unsupported CP fields type: {type(fields).__name__}")


def compare_fields(index, fields, verbose: bool = True, **kwargs) -> ComparisonReport:
    """
    Compare every CP field against the FA.

    The Retriever (and therefore the BM25 index) is built ONCE and reused across
    fields — rebuilding it per field is the dominant cost in a long CP form.

    Args:
        index:  a DocumentIndex* (or a Retriever, reused as-is).
        fields: dict {name: value}, list of query strings, list of
                {"field", "value"} dicts, or a path to a JSON file.
    """
    queries = _as_queries(fields)
    retriever = index if isinstance(index, Retriever) else Retriever(
        index,
        llm_model=kwargs.get("llm_model", "gemma3-27b-it"),
        base_url=kwargs.get("base_url", "http://localhost:11434/v1"),
        api_key=kwargs.get("api_key", "ollama"),
        use_llm=kwargs.get("use_llm", True),
        verbose=verbose,
    )

    results: list[FieldComparison] = []
    for i, query in enumerate(queries, start=1):
        if verbose:
            print(f"\n[{i}/{len(queries)}] {query[:100]}")
        try:
            results.append(compare_field(retriever, query, verbose=verbose, **kwargs))
        except Exception as exc:                        # noqa: BLE001
            print(f"      ⚠ Error: {type(exc).__name__}: {exc}", file=sys.stderr)
            results.append(FieldComparison(
                cp_field=query, deviation=False, needs_review=True,
                comments=f"[Error during processing: {type(exc).__name__}: {exc}]",
            ))
    return ComparisonReport(results=results)


def run_comparison(index, fields, verbose: bool = True, **kwargs) -> pd.DataFrame:
    """
    Compare every CP field and return the review sheet.

    Returns:
        pandas.DataFrame with columns
        S/N | CP Field | FA Legal Section | Deviation | Comments

    The structured working (per-item findings, cited passages, numeric flags) is
    available from compare_fields(), which this wraps.
    """
    return compare_fields(index, fields, verbose=verbose, **kwargs).to_dataframe()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="comparison",
        description="Compare CP fields against a facility agreement and write a "
                    "deviation review sheet.",
    )
    p.add_argument("--pdf", required=True, help="Path to the FA PDF.")
    p.add_argument("--cp", required=True,
                   help="JSON file: {field: value} or a list of query strings.")
    p.add_argument("--out", default="cp_review.xlsx",
                   help="Output .xlsx or .csv (default: cp_review.xlsx).")
    p.add_argument("--llm-model", default="gemma3-27b-it", metavar="MODEL")
    p.add_argument("--embedding-model", default="bge-m3", metavar="MODEL")
    p.add_argument("--base-url", default="http://localhost:11434/v1", metavar="URL")
    p.add_argument("--max-items", type=int, default=25, metavar="N",
                   help="Passages retrieved per field (default: 25).")
    p.add_argument("--min-grade", type=int, default=MIN_EVIDENCE_GRADE, metavar="G",
                   help="Retrieval grade a passage needs to be used (default: 2).")
    p.add_argument("--page-offset", type=int, default=0, metavar="N",
                   help="Added to page numbers in the FA Legal Section column.")
    p.add_argument("--include-heading", action="store_true",
                   help="Prepend the clause heading to each cited passage.")
    p.add_argument("--no-missing-deviation", action="store_true",
                   help="Do not treat an item the FA never addresses as a deviation.")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    from .ingestion_v3 import DocumentIndexV3

    print(f"Building document index from: {args.pdf}")
    index = DocumentIndexV3.from_pdf(
        pdf_path=args.pdf, llm_model=args.llm_model, base_url=args.base_url,
        api_key="ollama", embedding_model=args.embedding_model,
    )

    df = run_comparison(
        index, args.cp,
        llm_model=args.llm_model, base_url=args.base_url, api_key="ollama",
        max_items=args.max_items, min_evidence_grade=args.min_grade,
        page_offset=args.page_offset, include_heading=args.include_heading,
        treat_missing_as_deviation=not args.no_missing_deviation,
    )

    print("\n── Deviation Review Summary ─────────────────────────────────────")
    print(f"Fields checked  : {len(df)}")
    print(f"Deviations found: {(df['Deviation'] == 'Yes').sum()}")
    print("\n" + df[["S/N", "CP Field", "Deviation"]].to_string(index=False))

    if args.out.endswith(".csv"):
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(args.out, index=False, engine="openpyxl")
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
