"""
Single-field deviation check — the minimal Comparison Agent.

    llm   = ChatOpenAI(model=..., base_url=..., api_key=..., temperature=0)
    index = DocumentIndexV3.from_pdf("fa.pdf")
    res   = Retriever(index).retrieve(query)      # pipeline.retrieval
    row   = compare_one(query, res, llm)          # this module

`compare_one` takes the CP field (the query) and the RetrievalResult it produced,
and returns ONE row of the review sheet:

    S/N | CP Field | FA Legal Section | Deviation | Comments

Design notes
------------
Small on purpose — one LLM call, no batching, no CLI plumbing. What it does keep
is the part that makes the row trustworthy:

  • The agent is shown the ITEMS retrieval decomposed the query into, and is told
    to address every one of them. A CP field is rarely atomic ("USD 250m,
    quarterly repayment, Margin 2.25%"), and a free-text verdict otherwise
    answers the easy part and drops the rest.
  • The agent must declare which sections it USED. Only those land in the FA
    Legal Section cell, so the column is evidence rather than a dump of
    everything retrieval returned. Ids it invents are discarded.
  • Items retrieval could not cover are reported as not addressed in the FA —
    silence in the agreement is a finding, not a match.

The chat model is INJECTED (`llm`), not constructed here, so the caller owns the
endpoint, model and temperature — and the same client can be reused across
fields. If it is None, or the call fails, the row still carries its evidence and
coverage, marked for manual review rather than asserting "No deviation".
"""

import json
import re

import pandas as pd
from pydantic import BaseModel, Field

from .retrieval import RetrievalResult, RetrievedItem, Retriever

try:                                    # optional — only needed to call the model
    from langchain_core.messages import HumanMessage, SystemMessage
    _MESSAGES_AVAILABLE = True
except ImportError:                     # pragma: no cover - depends on install
    _MESSAGES_AVAILABLE = False

# ── Tunables ──────────────────────────────────────────────────────────────────

MIN_EVIDENCE_GRADE = 2      # retrieval grade a passage needs to reach the agent
MAX_SECTIONS = 8            # passages shown to the agent
SNIPPET_CHARS = 1200        # per-passage text budget in the prompt

# Endpoint defaults — used only by the demo main(); compare_one takes the client.
LLM_MODEL = "gemma3-27b-it"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"


# ── The row ───────────────────────────────────────────────────────────────────

class ComparisonRow(BaseModel):
    """One line item of the review sheet, plus the working behind it."""
    cp_field: str
    deviation: bool = False
    comments: str = ""
    sections: list[str] = Field(default_factory=list)   # formatted FA passages
    used_chunk_ids: list[int] = Field(default_factory=list)
    coverage: float = 0.0
    needs_review: bool = False

    @property
    def deviation_label(self) -> str:
        return "Yes" if self.deviation else "No"

    @property
    def fa_legal_section(self) -> str:
        return "\n\n".join(self.sections)

    def to_dict(self) -> dict:
        return {
            "CP Field": self.cp_field,
            "FA Legal Section": self.fa_legal_section,
            "Deviation": self.deviation_label,
            "Comments": self.comments,
        }


def to_dataframe(rows: list[ComparisonRow]) -> pd.DataFrame:
    """S/N | CP Field | FA Legal Section | Deviation | Comments."""
    df = pd.DataFrame([r.to_dict() for r in rows],
                      columns=["CP Field", "FA Legal Section", "Deviation", "Comments"])
    df.insert(0, "S/N", range(1, len(df) + 1))
    return df


# ── Formatting ────────────────────────────────────────────────────────────────

def _page_label(pages: list, offset: int = 0) -> str:
    """'(Page 33)' / '(Pages 33-34)' / '(Pages 33, 36)'."""
    nums = sorted({int(p) + offset for p in (pages or []) if str(p).lstrip("-").isdigit()})
    if not nums:
        return ""
    if len(nums) == 1:
        return f"(Page {nums[0]})"
    if nums == list(range(nums[0], nums[-1] + 1)):
        return f"(Pages {nums[0]}-{nums[-1]})"
    return "(Pages " + ", ".join(str(n) for n in nums) + ")"


def format_section(item: RetrievedItem, page_offset: int = 0) -> str:
    """`<content on one line> (Page N)` — the FA Legal Section cell format."""
    body = re.sub(r"\s+", " ", item.content.replace("\n", " ")).strip()
    return f"{body or item.heading.strip()} {_page_label(item.pages, page_offset)}".strip()


# ── Evidence ──────────────────────────────────────────────────────────────────

def select_evidence(
    res: RetrievalResult,
    min_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_SECTIONS,
) -> list[RetrievedItem]:
    """
    Narrow retrieval's output to what is worth showing the agent.

    Retrieval optimises recall — weak matches, definition supports, sibling
    parts. Here we want precision: confidently graded passages first, then
    supports (definitions the clauses lean on) with any leftover room.
    """
    primary = sorted((it for it in res.items
                      if not it.is_support and it.grade >= min_grade),
                     key=lambda it: (-it.grade, -it.score))
    chosen = primary[:max_sections]
    room = max_sections - len(chosen)
    if room > 0:
        chosen += [it for it in res.items if it.is_support][:room]
    return chosen


def _evidence_block(items: list[RetrievedItem]) -> str:
    """
    Render the passages for the prompt, carrying their retrieval metadata.

    The grade, the reranker's rationale and the items a passage was retrieved for
    are supplementary signal the agent would otherwise have to re-derive from the
    text — it is what separates "the operative clause for item 2" from "a
    definition that happens to share vocabulary".
    """
    blocks = []
    for it in items:
        meta = [f"pages {it.pages}"]
        if it.is_support:
            meta.append(f"supporting context via {it.via}")
        else:
            meta.append(f"retrieval grade {it.grade}/3")
            if it.covers:
                meta.append(f"retrieved for item(s) {it.covers}")
        if it.rationale:
            meta.append(f"relevance: {it.rationale}")
        blocks.append(
            f"### SECTION {it.chunk_id}\n"
            f"HEADING: {it.heading}\n"
            f"META: {'; '.join(meta)}\n"
            f"TEXT: {it.content.strip()[:SNIPPET_CHARS]}"
        )
    return "\n\n".join(blocks) if blocks else "(no passages retrieved)"


# ── The agent ─────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a legal analyst comparing a CP (conditions precedent / term sheet) "
    "field against the signed facility agreement (FA).\n\n"
    "You are given the CP FIELD, the ITEMS it breaks down into, and FA SECTIONS "
    "retrieved for it (each with an id).\n\n"
    "Decide whether the FA DEVIATES from the CP field. A deviation exists when the "
    "FA states something different (different figure, period, mechanism or "
    "condition), is more or less restrictive, or is SILENT on a required item. "
    "Judge only from the SECTIONS text — never assume the FA contains something "
    "you were not shown, and never treat silence as agreement.\n\n"
    "Return ONLY a JSON object:\n"
    '  {"deviation": true|false,\n'
    '   "used_sections": [<ids of the sections you actually relied on>],\n'
    '   "comments": "<explanation>"}\n\n'
    "The comments must address EVERY item listed, in order: for each, state what "
    "the FA actually says (quote the operative figure or wording) and whether it "
    "matches, deviates, or is not addressed. Cite the clause heading. Be concrete "
    "and concise — no preamble."
)


def _chat(llm, system: str, human: str, verbose: bool = True) -> str:
    """
    Invoke the injected chat model and return its text.

    Returns "" when there is no model or the call fails — the caller then takes
    the manual-review path instead of the whole row erroring out.
    """
    if llm is None:
        return ""
    try:
        messages = ([SystemMessage(content=system), HumanMessage(content=human)]
                    if _MESSAGES_AVAILABLE
                    else [("system", system), ("human", human)])
        out = llm.invoke(messages).content
        return out if isinstance(out, str) else str(out or "")
    except Exception as exc:                        # noqa: BLE001 — endpoint down, etc.
        if verbose:
            print(f"      → LLM call failed ({type(exc).__name__}: {exc})")
        return ""


def _parse_json_object(raw: str) -> dict:
    """Pull the JSON object out of the model's reply, tolerantly."""
    m = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# ── Main entry point ──────────────────────────────────────────────────────────

def compare_one(
    query: str,
    res: RetrievalResult,
    llm=None,
    min_evidence_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_SECTIONS,
    page_offset: int = 0,
    verbose: bool = True,
) -> ComparisonRow:
    """
    Compare one CP field against its retrieved FA passages.

    Args:
        query: the CP field — free text, may cover several requirements.
        res:   the RetrievalResult for that query (Retriever(index).retrieve(query)).
        llm:   a ChatOpenAI (or any chat model with .invoke(messages) -> .content).
               None runs the deterministic path: evidence and coverage only, the
               row marked for manual review.
        min_evidence_grade: retrieval grade a passage needs to reach the agent.
        page_offset: added to page numbers in the FA Legal Section cell (use the
                     index's page_offset to show printed page numbers).

    Returns:
        ComparisonRow — .deviation, .comments, .fa_legal_section, .to_dict().
    """
    evidence = select_evidence(res, min_evidence_grade, max_sections)
    by_id = {it.chunk_id: it for it in evidence}

    items_block = ("\n".join(f"  [{a.id}] {a.text}" for a in res.aspects)
                   or f"  [0] {query}")
    missing = [a.text for a in res.uncovered_aspects]

    missing_label = "; ".join(m[:80].rstrip(". ") for m in missing)

    human = (
        f"CP FIELD:\n{query}\n\n"
        f"ITEMS (address every one in your comments):\n{items_block}\n\n"
        + (f"ITEMS FOR WHICH RETRIEVAL FOUND NOTHING (treat as not addressed "
           f"unless a section below covers them):\n"
           + "\n".join(f"  - {m}" for m in missing) + "\n\n" if missing else "")
        + f"FA SECTIONS:\n{_evidence_block(evidence)}"
    )
    data = _parse_json_object(_chat(llm, _SYSTEM, human, verbose))

    # ── The agent's citations decide the evidence column ─────────────────────
    used_ids: list[int] = []
    for sid in data.get("used_sections") or []:
        try:
            sid = int(sid)
        except (TypeError, ValueError):
            continue
        if sid in by_id and sid not in used_ids:
            used_ids.append(sid)

    needs_review = not data
    comments = str(data.get("comments", "") or "").strip()
    deviation = bool(data.get("deviation", False))

    if needs_review:
        # No usable verdict — report the evidence and say so, rather than
        # recording a "No deviation" nobody stands behind.
        used_ids = [it.chunk_id for it in evidence if not it.is_support][:max_sections]
        comments = ("[Comparison agent unavailable — manual review required] "
                    f"Retrieval covered {res.coverage:.0%} of the field's items; "
                    f"{len(used_ids)} passage(s) retrieved.")
        if missing:
            comments += f" Not found in FA: {missing_label}."
    elif missing and not deviation:
        # The agent said no deviation while retrieval covered nothing for an
        # item — surface that rather than letting it pass silently.
        comments += f"\n⚠ No FA passage was found for: {missing_label}."

    row = ComparisonRow(
        cp_field=query,
        deviation=deviation,
        comments=comments,
        sections=[format_section(by_id[cid], page_offset) for cid in used_ids],
        used_chunk_ids=used_ids,
        coverage=res.coverage,
        needs_review=needs_review,
    )
    if verbose:
        print(f"      → Deviation: {row.deviation_label} · "
              f"{len(row.sections)} section(s) cited · coverage {res.coverage:.0%}")
    return row


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.comparison_one <pdf_path> <cp_field_text>")
        raise SystemExit(1)

    pdf_path, query = sys.argv[1], sys.argv[2]

    from langchain_openai import ChatOpenAI

    from .ingestion_v3 import DocumentIndexV3

    llm = ChatOpenAI(model=LLM_MODEL, base_url=BASE_URL, api_key=API_KEY,
                     temperature=0)

    index = DocumentIndexV3.from_pdf(pdf_path, llm_model=LLM_MODEL,
                                     base_url=BASE_URL, api_key=API_KEY)
    res = Retriever(index, llm_model=LLM_MODEL, base_url=BASE_URL,
                    api_key=API_KEY).retrieve(query)
    row = compare_one(query, res, llm, page_offset=index.page_offset)

    print("\n" + "=" * 70)
    print(res.coverage_report())
    print("=" * 70)
    print(f"CP Field  : {row.cp_field}")
    print(f"Deviation : {row.deviation_label}")
    print(f"Comments  :\n{row.comments}")
    print(f"\nFA Legal Section:\n{row.fa_legal_section}")

    df = to_dataframe([row])
    print("\n" + df.to_string(index=False, max_colwidth=60))


if __name__ == "__main__":
    main()
