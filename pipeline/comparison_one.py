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
    snapped to the document's own sentences and widened by CONTEXT_SENTENCES
    either side, so the FA Legal Section column shows the operative wording WITH
    the proviso or carve-out around it — enough for a reviewer to judge, without
    dumping the whole clause. Each entry ends with its section and page:

        …does not exceed 3.5:1. (21.2 Financial covenants, Page 30)
  • Items retrieval could not cover are reported as not addressed in the FA —
    silence in the agreement is a finding, not a match.
  • Items are fragments of the field, NOT independent statements. The agent is
    given the whole field first, the shared evidence pool for all items, and an
    explicit flag on any item that refers back ("the Company shall provide the
    above within 30 days", "such documents", "the foregoing"), which it must
    resolve against the earlier items before judging. A back-referencing item
    that retrieval could not cover is NEVER auto-marked "not addressed" — "the
    above" matches nothing in any agreement, so empty retrieval there says
    nothing about the FA. The agent also returns an `overall` verdict on the
    field read end to end, which can raise a deviation on its own.

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
MAX_QUOTE_SENTENCES = 3     # sentences the quote itself may span
CONTEXT_SENTENCES = 2       # sentences kept either side, so the quote reads in context
MAX_QUOTE_CHARS = 1400      # hard cap on one quoted passage in the sheet

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

# Wording that only means something in the context of what came BEFORE it:
# "the Company shall provide the above within 30 days", "such documents",
# "the foregoing". An item like that is not self-contained — retrieved alone it
# matches nothing useful, and judged alone it looks unaddressed.
_BACKREF_RE = re.compile(
    r"\b("
    r"the above|above[-\s]mentioned|abovementioned|aforementioned|aforesaid"
    r"|the foregoing|as (?:stated|set out|described|mentioned|listed) above"
    r"|the preceding|the said|said (?:documents?|items?|conditions?)"
    r"|such (?:documents?|items?|information|conditions?|requirements?|"
    r"deliverables?|matters?|amounts?|obligations?)"
    r"|there(?:of|in|to|under|with)"
    r"|the same"
    r")\b",
    re.IGNORECASE,
)
# An item that OPENS with a bare pronoun is continuing the previous sentence.
_LEADING_PRONOUN_RE = re.compile(
    r"^\s*(?:it|they|these|those|this|that|he|she|such)\b\s+(?:shall|must|will|is|are|was|were)",
    re.IGNORECASE,
)


def has_back_reference(text: str) -> bool:
    """
    True if the item leans on wording that came before it in the CP field.

    Used to keep such an item from being judged in isolation — both in what the
    agent is told about it, and in whether "retrieval found nothing" is allowed
    to become "the FA does not address this".
    """
    t = text or ""
    return bool(_BACKREF_RE.search(t) or _LEADING_PRONOUN_RE.match(t))


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
    heading: str = ""
    pages: list = Field(default_factory=list)
    position: float = 0.0      # where in the chunk it sits (0.0 start, 1.0 end)


class Finding(BaseModel):
    """The verdict on ONE item of the CP field."""
    item_id: int
    item_text: str
    deviation: bool = False
    explanation: str = ""
    quotes: list[Quote] = Field(default_factory=list)    # verbatim FA wording
    chunk_ids: list[int] = Field(default_factory=list)
    assessed: bool = True                                # False → manual review
    refers_back: bool = False                            # leans on earlier items

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
    overall_note: str = ""          # the agent's field-level observation

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

_PART_SUFFIX_RE = re.compile(r"\s*\(part\s*\d+\)\s*$", re.IGNORECASE)


def _page_numbers(pages: list, offset: int = 0) -> list[int]:
    return sorted({int(p) + offset for p in (pages or [])
                   if str(p).lstrip("-").isdigit()})


def page_for(pages: list, position: float = 0.0, offset: int = 0) -> int | None:
    """
    The single page a passage sits on.

    A chunk carries a LIST of pages (it may straddle a page break) but no
    per-line page map, so for a multi-page chunk the page is inferred from where
    the passage falls within it: `position` is the passage's relative offset
    (0.0 = start of the chunk, 1.0 = end). For a single-page chunk — the common
    case — the answer is exact.
    """
    nums = _page_numbers(pages, offset)
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]
    idx = int(max(0.0, min(0.999, position)) * len(nums))
    return nums[idx]


def section_label(heading: str) -> str:
    """The section name as a reviewer would cite it, without '(part N)' noise."""
    return _PART_SUFFIX_RE.sub("", _flatten(heading)).strip()


def format_section(text: str, heading: str, pages: list, position: float = 0.0,
                   page_offset: int = 0) -> str:
    """
    One FA Legal Section entry:

        <passage on one line> (<Section>, Page N)

    The passage ends with the section it came from and the page it is on, so a
    reviewer can go straight to it in the agreement.
    """
    body = _flatten(text)
    if len(body) > MAX_QUOTE_CHARS:
        body = body[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0] + "…"

    page = page_for(pages, position, page_offset)
    section = section_label(heading)
    if section and page is not None:
        return f"{body} ({section}, Page {page})"
    if section:
        return f"{body} ({section})"
    if page is not None:
        return f"{body} (Page {page})"
    return body


# ── Quote grounding: keep the sentence, drop the rest of the clause ───────────

def _with_context(sents: list[str], start: int, end: int,
                  context: int) -> tuple[str, float]:
    """
    Widen a sentence span by `context` sentences either side and return
    (passage, position) — position being where the span sits in the chunk
    (0.0 start, 1.0 end), which is what locates it on a page.
    """
    lo = max(0, start - context)
    hi = min(len(sents), end + context)
    passage = " ".join(sents[lo:hi]).strip()
    position = ((start + end) / 2.0) / max(1, len(sents))
    return passage, position


def snap_quote(quote: str, chunk_text: str,
               context: int = CONTEXT_SENTENCES) -> tuple[str, float] | None:
    """
    Locate `quote` in `chunk_text` and return the source sentences around it.

    The agent is asked for verbatim wording, but models paraphrase, re-case and
    re-punctuate. Rather than trust the quote or dump the whole clause, we find
    where it came from and return the document's OWN sentences — the ones
    carrying the quote, plus `context` sentences either side so the reviewer sees
    the proviso, the carve-out or the cross-reference that qualifies it.

    Returns (passage, position) or None when the quote is not in the passage at
    all (a hallucination, or a quote from a different section), so the caller can
    drop or replace it.
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
                return _with_context(sents, i, j + 1, context)

    # The quote spans more than the cap, or merges wording: keep the sentences
    # that are themselves inside the quote.
    hits = [i for i, s in enumerate(sents)
            if len(_norm(s)) >= 8 and _norm(s) in needle]
    if hits:
        return _with_context(sents, hits[0], hits[-1] + 1, context)
    return None


def best_sentences(chunk_text: str, target: str, max_sents: int = 2,
                   context: int = CONTEXT_SENTENCES) -> tuple[str, float]:
    """
    Deterministic fallback: the sentences of `chunk_text` overlapping `target`
    most, widened by `context` either side. Used when the agent quoted nothing
    usable, so the evidence column still shows the relevant part of the clause
    rather than the entire thing.
    """
    sents = _sentences(chunk_text)
    if not sents:
        return _flatten(chunk_text), 0.0
    target_tokens = set(re.findall(r"[a-z0-9][a-z0-9.:%,/-]*", _norm(target)))
    if not target_tokens:
        return _with_context(sents, 0, min(max_sents, len(sents)), context)

    scored = []
    for idx, sent in enumerate(sents):
        tokens = set(re.findall(r"[a-z0-9][a-z0-9.:%,/-]*", _norm(sent)))
        if not tokens:
            continue
        overlap = len(tokens & target_tokens) / len(target_tokens)
        scored.append((overlap, idx, sent))
    scored.sort(key=lambda s: (-s[0], s[1]))
    keep = sorted(idx for score, idx, _ in scored[:max_sents] if score > 0)
    if not keep:
        return _with_context(sents, 0, min(max_sents, len(sents)), context)
    return _with_context(sents, keep[0], keep[-1] + 1, context)


# ── Evidence ──────────────────────────────────────────────────────────────────

def select_evidence(
    res: RetrievalResult,
    min_grade: int = MIN_EVIDENCE_GRADE,
    max_sections: int = MAX_SECTIONS,
    include_weak: bool = True,
) -> list[RetrievedItem]:
    """
    Choose what the agent gets to see, best first:

      1. confidently graded passages (>= min_grade),
      2. weakly graded ones, if there is room,
      3. definition/cross-reference supports with whatever is left.

    Step 2 matters more than it looks. A grade is assigned PER ITEM, and an item
    that refers back ("the Company shall provide the above within 30 days")
    retrieves badly by construction — the clause that actually decides it (the
    delivery deadline) is scored against a fragment that names no subject, so it
    lands at grade 1 and a strict filter drops it. The agent then cannot cite the
    one clause it needed. Filling spare slots with weak passages costs nothing:
    they are labelled with their grade in the prompt, and the FA Legal Section
    column only ever shows passages the agent actually cited.

    Set include_weak=False for a strict, high-precision pool.
    """
    strong, weak, supports = [], [], []
    for it in res.items:
        if it.is_support:
            supports.append(it)
        elif it.grade >= min_grade:
            strong.append(it)
        elif it.grade >= 1:
            weak.append(it)

    strong.sort(key=lambda it: (-it.grade, -it.score))
    weak.sort(key=lambda it: (-it.grade, -it.score))

    chosen = strong[:max_sections]
    if include_weak and len(chosen) < max_sections:
        chosen += weak[: max_sections - len(chosen)]
    if len(chosen) < max_sections:
        chosen += supports[: max_sections - len(chosen)]
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
    "You are given the full CP FIELD, the ITEMS it breaks down into, and FA "
    "SECTIONS retrieved for the field as a whole (each with an id).\n\n"
    "READ THE WHOLE FIELD FIRST. The items are fragments of it, split for "
    "convenience — they are NOT independent. An item may only make sense in the "
    "context of the ones before it: 'the Company shall provide the above within "
    "30 days', 'such documents', 'the foregoing', 'these shall be certified'. "
    "Where an item refers back, RESOLVE it against the earlier items and the full "
    "field text before judging it, and say in your explanation what it resolves "
    "to. Items flagged (refers back) must never be judged in isolation.\n\n"
    "The SECTIONS were retrieved for the WHOLE field, so any item may be decided "
    "by any section — use whichever sections are relevant, not only the ones "
    "retrieved for that item.\n\n"
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
    "match the item. For an item that refers back, start by naming what it refers "
    "to. Max 40 words. Do NOT write a concluding 'therefore' sentence — that is "
    "added for you.\n\n"
    "Finally, judge the FIELD AS A WHOLE in \"overall\": anything that only "
    "appears when the field is read end to end — a requirement the items satisfy "
    "individually but the FA does not deliver together, or a back-reference the "
    "FA never picks up. Set its deviation to false when the per-item findings "
    "already say everything.\n\n"
    "Return ONLY a JSON object:\n"
    '  {"findings": [\n'
    '     {"item": <id>, "deviation": true|false, "sections": [<id>, ...],\n'
    '      "quotes": ["<verbatim FA wording>", ...],\n'
    '      "explanation": "<what the FA says and how it compares>"}\n'
    "   ],\n"
    '   "overall": {"deviation": true|false, "note": "<field-level observation, '
    'max 30 words, or empty>"}}'
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
            #
            # EXCEPT for an item that refers back to earlier ones: retrieval
            # searched for "the Company shall provide the above", which matches
            # nothing in any agreement. Empty retrieval there says nothing about
            # the FA, so calling it "not addressed" would invent a deviation.
            if not aspect.covered and has_back_reference(aspect.text):
                findings.append(Finding(
                    item_id=aspect.id, item_text=aspect.text, deviation=False,
                    explanation=("This item refers back to earlier items and could "
                                 "not be resolved on its own — read it with the "
                                 "items above; manual review required."),
                    assessed=False, refers_back=True,
                ))
            elif not aspect.covered:
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
                text, position = snapped
                key = _norm(text)
                if key not in seen:
                    seen.add(key)
                    quotes.append(Quote(text=text, chunk_id=sid,
                                        heading=by_id[sid].heading,
                                        pages=by_id[sid].pages, position=position))
                if sid not in ids:
                    ids.append(sid)
                break

        # A cited section that produced no usable quote still needs to show its
        # relevant wording — but the relevant part, not the whole clause.
        if not quotes:
            for sid in ids:
                snippet, position = best_sentences(by_id[sid].content,
                                                   f"{aspect.text} {query}")
                key = _norm(snippet)
                if snippet and key not in seen:
                    seen.add(key)
                    quotes.append(Quote(text=snippet, chunk_id=sid,
                                        heading=by_id[sid].heading,
                                        pages=by_id[sid].pages, position=position))

        findings.append(Finding(
            item_id=aspect.id, item_text=aspect.text,
            deviation=bool(entry.get("deviation", False)),
            explanation=str(entry.get("explanation", "") or "").strip(),
            quotes=quotes, chunk_ids=ids, assessed=True,
            refers_back=has_back_reference(aspect.text),
        ))

    return findings


def _compose_comments(findings: list[Finding], needs_review: bool,
                      overall_note: str = "") -> str:
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

    if overall_note:
        lines.append(f"Overall: {overall_note.rstrip('.')}.")
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

    # Items that lean on earlier ones are flagged for the agent, so it resolves
    # them against the field instead of judging a fragment.
    backref_ids = {a.id for a in res.aspects if has_back_reference(a.text)}
    items_block = ("\n".join(
        f"  [{a.id}] {a.text}"
        + ("  (refers back — resolve against the items above and the CP FIELD)"
           if a.id in backref_ids else "")
        for a in res.aspects) or f"  [0] {query}")

    # A back-referencing item retrieves nothing because "the above" matches
    # nothing — that is a property of the query, not of the FA. Listing it as
    # "found nothing" would push the agent to call it unaddressed.
    missing = [a.text for a in res.uncovered_aspects if a.id not in backref_ids]

    human = (
        f"CP FIELD (read this in full first — the items below are fragments of "
        f"it):\n{query}\n\n"
        f"ITEMS (return one finding for EACH):\n{items_block}\n\n"
        + (f"ITEMS FOR WHICH RETRIEVAL FOUND NOTHING (treat as not addressed "
           f"unless a section below covers them):\n"
           + "\n".join(f"  - {m}" for m in missing) + "\n\n" if missing else "")
        + f"FA SECTIONS (retrieved for the whole field — any item may be decided "
          f"by any of them):\n{_evidence_block(evidence)}"
    )
    data = _parse_json_object(_chat(llm, _SYSTEM, human, verbose))

    findings = _build_findings(data, res, by_id, query, treat_missing_as_deviation)
    needs_review = any(not f.assessed for f in findings) or not findings

    # ── The field read as a whole, on top of the per-item findings ──────────
    overall = data.get("overall") if isinstance(data.get("overall"), dict) else {}
    overall_deviation = bool(overall.get("deviation", False))
    overall_note = str(overall.get("note", "") or "").strip()

    # ── The row's verdict: Yes if ANY item deviates, No only if none do ──────
    deviation = any(f.deviation for f in findings if f.assessed) or overall_deviation

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
            sections.append(format_section(quote.text, quote.heading, quote.pages,
                                           quote.position, page_offset))

    if not findings:
        # Retrieval produced no items at all — nothing to compare.
        comments = ("[No items could be derived from this field — manual review "
                    "required]")
    else:
        comments = _compose_comments(findings, needs_review, overall_note)

    row = ComparisonRow(
        cp_field=query, deviation=deviation, comments=comments, sections=sections,
        findings=findings, used_chunk_ids=used_ids, coverage=res.coverage,
        needs_review=needs_review, overall_note=overall_note,
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
    res = Retriever(index, llm).retrieve(query)
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
