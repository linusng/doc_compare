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

  • The agent works PER ITEM. retrieval.py has already decomposed the field into
    its individual requirements; the agent returns a finding for each, so a
    multi-part field ("USD 250m, quarterly repayment, Margin 2.25%") cannot have
    its awkward parts quietly skipped. Each item's comment ends in the explicit
    sentence "Therefore, it is (not) a deviation."
  • The row's Deviation is Yes if ANY item deviates, No only when every item is
    clean.
  • The agent must QUOTE the wording it relied on, and every quote is checked
    verbatim against the retrieved text before it reaches the sheet. Quotes are
    snapped to the sentences they came from, so the FA Legal Section column holds
    the operative sentence — not the whole clause — and stays reviewable.
  • Items retrieval could not cover are reported as not addressed in the FA —
    silence in the agreement is a finding, not a match.

The chat model is INJECTED (`llm`), not constructed here, so the caller owns the
endpoint, model and temperature — and the same client can be reused across
fields. If it is None, or the call fails, the row still carries its evidence and
coverage, marked for manual review rather than asserting "No deviation".
"""

import json
import re
import unicodedata

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
MAX_QUOTE_SENTENCES = 3     # a quote is snapped to at most this many sentences
MAX_QUOTE_CHARS = 450       # hard cap on one quoted passage in the sheet

# The mandated closing sentences — every assessed item ends with one of these.
DEVIATION_SENTENCE = "Therefore, it is a deviation."
NO_DEVIATION_SENTENCE = "Therefore, it is not a deviation."

# Endpoint defaults — used only by the demo main(); compare_one takes the client.
LLM_MODEL = "gemma3-27b-it"
BASE_URL = "http://localhost:11434/v1"
API_KEY = "ollama"


# ── Text normalisation (for verbatim grounding) ───────────────────────────────

_PUNCT_TABLE = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "″": '"',
    "‐": "-", "‑": "-", "–": "-", "—": "-", "−": "-",
    " ": " ", "​": "", "­": "",
})


def _norm(text: str) -> str:
    """Fold cosmetic differences only — smart quotes, dashes, spacing, case."""
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ")).strip()


_SENTENCE_END_RE = re.compile(r"(?<=[.;:])\s+")


def _sentences(text: str) -> list[str]:
    """Split a passage into sentence-ish units, flattened to single lines."""
    parts = []
    for line in (text or "").split("\n"):
        for sent in _SENTENCE_END_RE.split(line):
            sent = sent.strip()
            if sent:
                parts.append(sent)
    return parts


# ── The row ───────────────────────────────────────────────────────────────────

class Quote(BaseModel):
    """A verbatim passage from the FA, tied to the section it came from."""
    text: str
    chunk_id: int
    pages: list = Field(default_factory=list)


class Finding(BaseModel):
    """The verdict on ONE item of the CP field."""
    item_id: int
    item_text: str
    deviation: bool = False
    explanation: str = ""
    quotes: list[Quote] = Field(default_factory=list)    # verbatim FA wording
    chunk_ids: list[int] = Field(default_factory=list)
    assessed: bool = True                                # False → manual review

    def comment(self) -> str:
        """`<item>: <explanation> Therefore, it is (not) a deviation.`"""
        body = (self.explanation or "").strip()
        # Drop any closing sentence the model wrote itself, so the mandated one
        # is the single source of truth.
        body = re.sub(r"\s*therefore,?\s+it\s+is\s+(?:not\s+)?a\s+deviation\.?\s*$",
                      "", body, flags=re.IGNORECASE).strip()
        if body and not body.endswith((".", "!", "?")):
            body += "."
        head = f"[{self.item_id + 1}] {self.item_text.rstrip('.')}"
        if not self.assessed:
            return f"{head} — {body or 'Not assessed by the comparison agent.'}"
        tail = DEVIATION_SENTENCE if self.deviation else NO_DEVIATION_SENTENCE
        return f"{head} — {body} {tail}".replace("  ", " ").strip()


class ComparisonRow(BaseModel):
    """One line item of the review sheet, plus the working behind it."""
    cp_field: str
    deviation: bool = False
    comments: str = ""
    sections: list[str] = Field(default_factory=list)   # formatted FA quotes
    findings: list[Finding] = Field(default_factory=list)
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


def format_section(text: str, pages: list, page_offset: int = 0) -> str:
    """`<passage on one line> (Page N)` — the FA Legal Section cell format."""
    body = _flatten(text)
    if len(body) > MAX_QUOTE_CHARS:
        body = body[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0] + "…"
    return f"{body} {_page_label(pages, page_offset)}".strip()


# ── Quote grounding: keep the sentence, drop the rest of the clause ───────────

def snap_quote(quote: str, chunk_text: str) -> str | None:
    """
    Locate `quote` in `chunk_text` and return the SOURCE sentence(s) containing it.

    The agent is asked for verbatim wording, but models paraphrase, re-case and
    re-punctuate. Rather than trust the quote or dump the whole clause, we find
    where it came from and return the document's own sentences — so the sheet
    shows real FA text, trimmed to the part that was actually compared.

    Returns None when the quote is not in the passage at all (a hallucination, or
    a quote from a different section), so the caller can drop or replace it.
    """
    needle = _norm(quote)
    if len(needle) < 8:
        return None

    sents = _sentences(chunk_text)
    if not sents:
        return None

    # A run of consecutive sentences that CONTAINS the quote.
    for i in range(len(sents)):
        acc = ""
        for j in range(i, min(i + MAX_QUOTE_SENTENCES, len(sents))):
            acc = f"{acc} {sents[j]}".strip()
            if needle in _norm(acc):
                return acc

    # The quote spans more than the cap, or merges wording: keep the sentences
    # that are themselves inside the quote.
    hits = [s for s in sents if len(_norm(s)) >= 8 and _norm(s) in needle]
    if hits:
        return " ".join(hits[:MAX_QUOTE_SENTENCES])
    return None


def best_sentences(chunk_text: str, target: str, max_sents: int = 2) -> str:
    """
    Deterministic fallback: the sentences of `chunk_text` that overlap `target`
    most. Used when the agent quoted nothing usable, so the evidence column still
    shows the relevant sentence rather than the entire clause.
    """
    sents = _sentences(chunk_text)
    if not sents:
        return _flatten(chunk_text)
    target_tokens = set(re.findall(r"[a-z0-9][a-z0-9.:%,/-]*", _norm(target)))
    if not target_tokens:
        return " ".join(sents[:max_sents])

    scored = []
    for idx, sent in enumerate(sents):
        tokens = set(re.findall(r"[a-z0-9][a-z0-9.:%,/-]*", _norm(sent)))
        if not tokens:
            continue
        overlap = len(tokens & target_tokens) / len(target_tokens)
        scored.append((overlap, idx, sent))
    scored.sort(key=lambda s: (-s[0], s[1]))
    keep = [s for s in scored[:max_sents] if s[0] > 0]
    if not keep:
        return " ".join(sents[:max_sents])
    return " ".join(sent for _, _, sent in sorted(keep, key=lambda s: s[1]))


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
    "Return ONE finding per item. For each item decide whether the FA DEVIATES "
    "from it. A deviation exists when the FA states something different (a "
    "different figure, period, mechanism or condition), is more or less "
    "restrictive, or is SILENT on the item.\n\n"
    "Rules:\n"
    "- Judge ONLY from the SECTIONS text. Never assume the FA contains something "
    "you were not shown, and never treat silence as agreement.\n"
    "- Compare figures and periods EXACTLY: 2.25% vs 2.50%, quarterly vs "
    "semi-annually, and USD vs EUR are deviations.\n"
    "- quotes: copy the SHORTEST wording from the SECTIONS that decides the item "
    "— the operative sentence, character-for-character. Do not paraphrase, do not "
    "quote a whole clause, and do not quote text you were not shown.\n"
    "- sections: the ids you actually relied on.\n"
    "- explanation: state what the FA actually says and why it does or does not "
    "match the item. Max 35 words. Do NOT write a concluding 'therefore' "
    "sentence — that is added for you.\n\n"
    "Return ONLY a JSON object:\n"
    '  {"findings": [\n'
    '     {"item": <id>, "deviation": true|false, "sections": [<id>, ...],\n'
    '      "quotes": ["<verbatim FA wording>", ...],\n'
    '      "explanation": "<what the FA says and how it compares>"}\n'
    "  ]}"
)


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


# ── Finding assembly ──────────────────────────────────────────────────────────

def _build_findings(
    data: dict,
    res: RetrievalResult,
    by_id: dict[int, RetrievedItem],
    query: str,
    treat_missing_as_deviation: bool,
) -> list[Finding]:
    """
    Turn the agent's reply into exactly one finding per item.

    Guarantees enforced here, because the model gets each of them wrong often
    enough to matter:
      • every item gets a finding — skipped ones fall back to retrieval coverage;
      • only real section ids survive;
      • only quotes that ground verbatim in a cited section survive, snapped to
        the source sentence. A cited section with no usable quote falls back to
        its most relevant sentences rather than the whole clause.
    """
    raw = data.get("findings")
    by_item: dict[int, dict] = {}
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                by_item.setdefault(int(entry.get("item")), entry)
            except (TypeError, ValueError):
                continue

    aspects = res.aspects or []
    findings: list[Finding] = []

    for aspect in aspects:
        entry = by_item.get(aspect.id)

        if entry is None:
            # Retrieval found nothing → the FA is silent, which is a finding we
            # can make without the model. Otherwise it is simply unassessed.
            if not aspect.covered:
                findings.append(Finding(
                    item_id=aspect.id, item_text=aspect.text,
                    deviation=bool(treat_missing_as_deviation),
                    explanation="No provision addressing this item was found in the FA.",
                    assessed=True,
                ))
            else:
                findings.append(Finding(
                    item_id=aspect.id, item_text=aspect.text, deviation=False,
                    explanation=("Passages were retrieved for this item but the "
                                 "comparison agent returned no finding — manual "
                                 "review required."),
                    assessed=False,
                    chunk_ids=[c for c in aspect.evidence if c in by_id][:2],
                ))
            continue

        ids: list[int] = []
        for sid in entry.get("sections") or []:
            try:
                sid = int(sid)
            except (TypeError, ValueError):
                continue
            if sid in by_id and sid not in ids:
                ids.append(sid)

        # Ground each quote against the sections the agent cited (and, failing
        # that, any retrieved section — models mislabel the id more often than
        # they invent the text).
        quotes: list[Quote] = []
        seen: set[str] = set()
        for quote in entry.get("quotes") or []:
            quote = str(quote or "").strip()
            if not quote:
                continue
            search_ids = ids or list(by_id)
            for sid in list(search_ids) + [i for i in by_id if i not in search_ids]:
                snapped = snap_quote(quote, by_id[sid].content)
                if snapped is None:
                    continue
                key = _norm(snapped)
                if key not in seen:
                    seen.add(key)
                    quotes.append(Quote(text=snapped, chunk_id=sid,
                                        pages=by_id[sid].pages))
                if sid not in ids:
                    ids.append(sid)
                break

        # A cited section that produced no usable quote still needs to show its
        # relevant wording — but sentence-level, not the whole clause.
        if not quotes:
            for sid in ids:
                snippet = best_sentences(by_id[sid].content, f"{aspect.text} {query}")
                key = _norm(snippet)
                if snippet and key not in seen:
                    seen.add(key)
                    quotes.append(Quote(text=snippet, chunk_id=sid,
                                        pages=by_id[sid].pages))

        findings.append(Finding(
            item_id=aspect.id, item_text=aspect.text,
            deviation=bool(entry.get("deviation", False)),
            explanation=str(entry.get("explanation", "") or "").strip(),
            quotes=quotes, chunk_ids=ids, assessed=True,
        ))

    return findings


def _compose_comments(findings: list[Finding], needs_review: bool) -> str:
    """
    One line per item, IN ITEM ORDER, each ending in the mandated sentence.

    Assembling this deterministically (rather than asking the model for a
    paragraph) is what guarantees the whole field is addressed: a model writing
    prose drops the uninteresting items. Item order is kept so the cell reads
    against the CP field itself; the header line carries the deviation count so
    nothing important depends on reading to the end.
    """
    if not findings:
        return ""

    deviating = [f for f in findings if f.assessed and f.deviation]
    clean = [f for f in findings if f.assessed and not f.deviation]
    unassessed = [f for f in findings if not f.assessed]

    head = []
    if deviating:
        head.append(f"{len(deviating)} deviation(s)")
    if clean:
        head.append(f"{len(clean)} no deviation")
    if unassessed:
        head.append(f"{len(unassessed)} not assessed")
    lines = [f"{len(findings)} item(s) checked: " + ", ".join(head) + "."]

    lines.extend(f.comment() for f in sorted(findings, key=lambda f: f.item_id))

    if needs_review:
        lines.append("⚠ Manual review required — see the items marked not assessed.")
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────

def compare_one(
    query: str,
    res: RetrievalResult,
    llm=None,
    min_evidence_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_SECTIONS,
    page_offset: int = 0,
    treat_missing_as_deviation: bool = True,
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
        treat_missing_as_deviation: an item no FA passage addresses counts as a
                     deviation (default True — FA silence is a review finding).

    Returns:
        ComparisonRow. `.deviation` is True if ANY item deviates; `.comments`
        carries one line per item, each ending "Therefore, it is (not) a
        deviation."; `.fa_legal_section` holds the quoted FA sentences with pages.
    """
    evidence = select_evidence(res, min_evidence_grade, max_sections)
    by_id = {it.chunk_id: it for it in evidence}

    items_block = ("\n".join(f"  [{a.id}] {a.text}" for a in res.aspects)
                   or f"  [0] {query}")
    missing = [a.text for a in res.uncovered_aspects]

    human = (
        f"CP FIELD:\n{query}\n\n"
        f"ITEMS (return one finding for EACH):\n{items_block}\n\n"
        + (f"ITEMS FOR WHICH RETRIEVAL FOUND NOTHING (treat as not addressed "
           f"unless a section below covers them):\n"
           + "\n".join(f"  - {m}" for m in missing) + "\n\n" if missing else "")
        + f"FA SECTIONS:\n{_evidence_block(evidence)}"
    )
    data = _parse_json_object(_chat(llm, _SYSTEM, human, verbose))

    findings = _build_findings(data, res, by_id, query, treat_missing_as_deviation)
    needs_review = any(not f.assessed for f in findings) or not findings

    # ── The row's verdict: Yes if ANY item deviates, No only if none do ──────
    deviation = any(f.deviation for f in findings if f.assessed)

    # ── FA Legal Section: the quoted sentences, in item order, deduped ───────
    sections: list[str] = []
    used_ids: list[int] = []
    seen: set[str] = set()
    for f in findings:
        for cid in f.chunk_ids:
            if cid not in used_ids:
                used_ids.append(cid)
        for quote in f.quotes:
            key = _norm(quote.text)
            if key in seen:
                continue
            seen.add(key)
            sections.append(format_section(quote.text, quote.pages, page_offset))

    if not findings:
        # Retrieval produced no items at all — nothing to compare.
        comments = ("[No items could be derived from this field — manual review "
                    "required]")
    else:
        comments = _compose_comments(findings, needs_review)

    row = ComparisonRow(
        cp_field=query, deviation=deviation, comments=comments, sections=sections,
        findings=findings, used_chunk_ids=used_ids, coverage=res.coverage,
        needs_review=needs_review,
    )
    if verbose:
        n_dev = sum(1 for f in findings if f.assessed and f.deviation)
        print(f"      → Deviation: {row.deviation_label} · "
              f"{n_dev}/{len(findings)} item(s) deviate · "
              f"{len(sections)} passage(s) quoted · coverage {res.coverage:.0%}")
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
