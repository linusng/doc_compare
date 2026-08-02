"""
Coverage-driven agentic retrieval — the retrieval front-end for long, multi-item
free-text inputs.

Main entry point: retrieve(index, text, ...) -> RetrievalResult (a list of items)

The problem this solves
-----------------------
The rest of the pipeline retrieves for a SHORT input: a section name, a
description of one passage, one question. This module is built for the opposite
case — the input is a large free-text field that talks about SEVERAL different
things at once, each of which lives in a DIFFERENT part of the agreement:

    "The Facility is a USD 250,000,000 term loan repayable in equal quarterly
     instalments. Interest accrues at Term SOFR plus the Margin. The Borrower
     must maintain a Leverage Ratio of not more than 3.5:1 tested semi-annually,
     and may not dispose of assets exceeding 10% of consolidated total assets in
     any financial year without Majority Lender consent (see Clause 22.4)."

Embedding that whole paragraph as ONE query is the classic failure mode: the
resulting vector is an average of five unrelated topics and lands squarely
between all of them, so the amount clause, the interest clause, the financial
covenant and the disposals covenant are each retrieved weakly, or not at all —
and a single similarity ranking has no way to tell you which of them you MISSED.

The method
----------
Retrieval is organised around COVERAGE of the input rather than similarity to
it. Each stage exists to fix a specific failure of naive top-k RAG:

  1. DECOMPOSE   The input is broken into atomic *aspects* — one retrievable
                 item each (a deterministic segmenter, refined by an LLM when
                 one is available). Every aspect gets its own retrieval budget,
                 so a short clause mentioned once in a long paragraph is never
                 drowned out by the paragraph's bulk.
  2. PLAN        Per aspect: the aspect text, its key terms (quoted terms,
                 capitalised defined terms, acronyms, amounts, ratios), explicit
                 cross-references, a defined-term probe ('"Margin" means'), and
                 optionally a HyDE clause — a hypothetical piece of contract
                 language, which embeds far closer to the real clause than a
                 description of it does.
  3. HYBRID      Every query runs against BOTH a dense vector search and a
                 sparse BM25F index built here from scratch (headings weighted).
                 Dense catches paraphrase, sparse catches exact identifiers
                 ("3.5:1", "Clause 22.4", "SOFR") that embeddings routinely
                 blur. Deterministic lexical lookups (definition regex, section
                 number, exact phrase) are folded in as a third channel.
  4. FUSE        Rankings are combined with Reciprocal Rank Fusion, which needs
                 no score calibration between channels — the standard fix for
                 "BM25 scores and cosine scores are not comparable".
  5. RERANK      Candidates are graded 0-3 against their aspect by an LLM
                 (batched, one call per aspect) — a cheap cross-encoder stand-in
                 that fixes the recall/precision gap of pure vector ranking. A
                 lexical grader takes over verbatim when no LLM is reachable.
  6. AGENTIC     Aspects with no grade >= min_grade are UNCOVERED. The retriever
     REFINE      then behaves like an agent: it shows the LLM what it retrieved,
                 asks for genuinely different search angles (synonyms, the likely
                 defined term, the operative clause it would sit in), and runs
                 another round — repeating up to `max_rounds`. This is the loop
                 that turns "I found 8 chunks" into "I found something for every
                 item, and here is what I could not find".
  7. DIVERSIFY   Within an aspect, near-duplicate chunks are suppressed with MMR
                 (over the vectors already in the store — no extra embedding
                 calls), so slots are not spent on five copies of one clause.
  8. SELECT      Items are picked ROUND-ROBIN across aspects, best-first. This is
                 what makes the output cover the input: a strong aspect cannot
                 consume the whole budget while another aspect goes unrepresented.
  9. EXPAND      Finally the definition graph is walked (defined terms and
                 section cross-references found in the selected text) so the
                 returned set is self-contained — a clause that turns on
                 "Permitted Disposal" is useless without that definition.

Everything degrades gracefully: with no LLM endpoint reachable the module still
runs end-to-end on deterministic segmentation, BM25F + lexical lookups and a
lexical grader; with no vector store it runs sparse-only. No stage is required
for the pipeline to produce an answer.

Standalone
----------
This module imports NOTHING from the rest of the pipeline — tokenisation,
normalisation, section-number parsing, the sparse index, fusion, reranking and
the definition graph are all implemented here. It only duck-types the index:
any object with `.chunks` (objects with heading/content/pages/chunk_id, or
dicts) and, optionally, `.vector_store` works — DocumentIndex, DocumentIndexV2
and DocumentIndexV3 all satisfy that.

Usage
-----
    from pipeline.ingestion_v3 import DocumentIndexV3
    from pipeline.retrieval import retrieve, Retriever

    index = DocumentIndexV3.from_pdf("agreement.pdf")

    result = retrieve(index, long_free_text)

    for item in result:                       # the result IS a list of items
        print(item.heading, item.pages, item.grade, item.covers)

    print(result.coverage_report())           # what was found / missed
    context = result.combined_context         # ready for the next LLM step

    # Reuse across many queries (BM25 index built once):
    r = Retriever(index)
    items_a = r.retrieve(text_a).items
    items_b = r.retrieve(text_b).items
"""

import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

try:                                    # optional — the module runs without them
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    _LANGCHAIN_AVAILABLE = True
except ImportError:                     # pragma: no cover - depends on install
    _LANGCHAIN_AVAILABLE = False


# ── Tunables ──────────────────────────────────────────────────────────────────

DEFAULT_LLM_MODEL = "gemma3-27b-it"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_API_KEY = "ollama"

MAX_ASPECTS = 24            # hard cap on items decomposed out of the input
TOP_K_PER_QUERY = 8         # candidates pulled per query, per channel
CANDIDATES_PER_ASPECT = 12  # candidates that survive fusion into reranking
MAX_ITEMS = 25              # primary (aspect-covering) items returned
MAX_SUPPORT_ITEMS = 10      # extra definition / cross-reference items
MAX_ROUNDS = 3              # retrieve → grade → refine rounds (1 = no refinement)
MIN_GRADE = 2               # grade (0-3) at which an aspect counts as covered
MMR_LAMBDA = 0.7            # 1.0 = pure relevance, 0.0 = pure diversity
RRF_K = 60                  # Reciprocal Rank Fusion damping constant
EXPAND_DEPTH = 1            # definition-graph hops from the selected items
MAX_LLM_CALLS = 60          # budget guard; past this the module goes heuristic

# Channel weights used when fusing rankings (RRF contributions are multiplied).
CHANNEL_WEIGHTS = {
    "dense": 1.0,       # embeddings — paraphrase / semantic match
    "sparse": 0.95,     # BM25F — exact identifiers, numbers, rare words
    "lexical": 1.25,    # definition / section-number / exact-phrase hits
}


# ── Normalisation & tokenisation ──────────────────────────────────────────────
#
# Legal PDFs are full of typographic variants that break naive string matching:
# smart quotes, en/em dashes, ligatures, soft hyphens, non-breaking spaces. Both
# sides of every comparison in this module are pushed through _normalize, so only
# cosmetic differences are forgiven — never paraphrase.

_PUNCT_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'", "`": "'", "´": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"', "«": '"', "»": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-", "−": "-",
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "",
    "…": "...",
}
_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)


def _normalize(text: str) -> str:
    """NFKC-fold, unify punctuation variants, collapse whitespace, lowercase."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip().lower()


# Tokens keep internal dots, slashes, colons, hyphens and percent signs so that
# legal identifiers survive intact: "3.5:1", "22.4", "10%", "usd250,000,000",
# "sofr". Splitting those would destroy exactly the signal sparse search is for.
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9.:/%,'-]*")

_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "but",
    "by", "can", "could", "did", "do", "does", "each", "for", "from", "had",
    "has", "have", "how", "if", "in", "into", "is", "it", "its", "may", "must",
    "no", "not", "of", "on", "or", "other", "shall", "should", "so", "such",
    "than", "that", "the", "their", "then", "there", "these", "this", "those",
    "to", "under", "until", "was", "were", "what", "when", "where", "which",
    "while", "who", "whom", "whose", "why", "will", "with", "would",
}


def _stem(token: str) -> str:
    """
    Very light suffix stripping — enough to unify 'disposals'/'disposal' and
    'repayments'/'repayment' without the false conflations a real stemmer makes
    on legal vocabulary (we never want 'interest'/'interested' merged away).
    """
    t = token
    if len(t) > 4:
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("sses"):
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
    if len(t) > 6 and t.endswith("ing"):
        return t[:-3]
    if len(t) > 5 and t.endswith("ed"):
        return t[:-2]
    return t


def _tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    """Normalise → tokenise → drop stopwords → light stem."""
    toks = _TOKEN_RE.findall(_normalize(text))
    out = []
    for t in toks:
        t = t.strip(".,:;'-/")
        if not t or (not keep_stopwords and t in _STOPWORDS):
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        out.append(_stem(t))
    return out


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Chunk adaptation (duck-typed index) ───────────────────────────────────────

@dataclass
class _ChunkView:
    """A pipeline chunk, normalised into the shape this module needs."""
    chunk_id: int
    heading: str
    content: str
    pages: list
    order: int                       # position in document order
    tokens: list = field(default_factory=list)
    token_set: set = field(default_factory=set)
    norm_text: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


def _get(obj, name, default=None):
    """Read `name` off an object or a dict."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _adapt_chunks(index) -> list[_ChunkView]:
    """
    Build _ChunkViews from any index exposing `.chunks` (SectionChunk objects,
    plain objects or dicts). Missing chunk_ids fall back to document order.
    """
    raw = _get(index, "chunks", None)
    if raw is None:
        raise ValueError("index has no `.chunks` — cannot retrieve")

    views: list[_ChunkView] = []
    for i, c in enumerate(raw):
        heading = str(_get(c, "heading", "") or "")
        content = str(_get(c, "content", "") or "")
        if not content and not heading:
            continue
        cid = _get(c, "chunk_id", None)
        try:
            cid = int(cid) if cid is not None else i
        except (TypeError, ValueError):
            cid = i
        pages = _get(c, "pages", None) or []
        view = _ChunkView(
            chunk_id=cid, heading=heading, content=content,
            pages=list(pages), order=i,
        )
        view.tokens = _tokenize(view.full_text)
        view.token_set = set(view.tokens)
        view.norm_text = _normalize(view.full_text)
        views.append(view)
    return views


# ── Sparse retrieval: BM25F built from scratch ────────────────────────────────

class BM25Index:
    """
    BM25F over the document's chunks, with the heading treated as a boosted
    field.

    Why this exists at all: the pipeline is dense-only today, and embeddings are
    reliably WEAK on precisely the tokens legal retrieval turns on — clause
    numbers ("22.4"), ratios ("3.5:1"), amounts, currency codes, party names and
    rare defined terms. BM25 nails those. Running both and fusing is strictly
    better than either alone, and costs nothing at query time.

    Standard Okapi BM25 with a field-weighted term frequency:
        score(q, d) = Σ_t idf(t) · (tf'(t,d) · (k1+1)) /
                                   (tf'(t,d) + k1 · (1 - b + b · dl'/avgdl'))
    where tf' counts a heading occurrence `heading_weight` times, and dl' is the
    matching weighted length (so boosting the heading cannot silently inflate
    long documents' scores).
    """

    def __init__(
        self,
        chunks: list[_ChunkView],
        k1: float = 1.5,
        b: float = 0.75,
        heading_weight: float = 3.0,
    ):
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.n_docs = len(chunks)

        self._tf: list[dict[str, float]] = []
        self._len: list[float] = []
        self._postings: dict[str, list[int]] = defaultdict(list)

        for i, ch in enumerate(chunks):
            body_tokens = _tokenize(ch.content)
            head_tokens = _tokenize(ch.heading)
            tf: dict[str, float] = Counter(body_tokens)
            for t in head_tokens:
                tf[t] = tf.get(t, 0.0) + heading_weight
            self._tf.append(dict(tf))
            self._len.append(float(len(body_tokens)) + heading_weight * len(head_tokens))
            for t in tf:
                self._postings[t].append(i)

        self.avg_len = (sum(self._len) / self.n_docs) if self.n_docs else 0.0

        # Robertson/Sparck-Jones idf with the +1 smoothing that keeps it positive
        # for terms appearing in more than half the corpus (common in one-document
        # corpora, where a term like "borrower" is everywhere).
        self.idf: dict[str, float] = {}
        for term, docs in self._postings.items():
            df = len(docs)
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = TOP_K_PER_QUERY) -> list[tuple[int, float]]:
        """Return [(chunk_index, score)] best-first. Indices are into `chunks`."""
        q_tokens = _tokenize(query)
        if not q_tokens or not self.n_docs:
            return []

        scores: dict[int, float] = defaultdict(float)
        # Repeated query terms should count once per occurrence, but a query that
        # repeats a term 20 times must not dominate — cap the multiplier at 3.
        for term, q_tf in Counter(q_tokens).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            q_weight = min(float(q_tf), 3.0)
            for i in self._postings[term]:
                tf = self._tf[i].get(term, 0.0)
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * (self._len[i] / (self.avg_len or 1.0))
                )
                scores[i] += q_weight * idf * (tf * (self.k1 + 1.0)) / (denom or 1.0)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        return ranked[:top_k]


# ── Deterministic lexical lookups (definitions, cross-refs, exact phrases) ─────

_LEADING_LABEL_RE = re.compile(
    r"^(?:(?:sections?|articles?|clauses?|paragraphs?|sec|art)\b|§)[\s.\-:]*",
    re.IGNORECASE,
)
_ROMAN_RE = re.compile(
    r"^(M{0,4}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3}))\b",
    re.IGNORECASE,
)
_NAMED_PREFIX_RE = re.compile(
    r"^(schedule|annex|appendix|exhibit|attachment|part)\s+(\d+(?:\.\d+)*|[a-z0-9]+)",
    re.IGNORECASE,
)
_PART_SUFFIX_RE = re.compile(r"\s*\(part \d+\)\s*$", re.IGNORECASE)
_MAX_ROMAN_SECTION = 49


def _roman_to_int(s: str) -> int:
    vals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total, prev = 0, 0
    for ch in reversed(s.lower()):
        v = vals[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


def _base_heading(heading: str) -> str:
    """'1.1 Definitions (part 2)' → '1.1 Definitions'."""
    return _PART_SUFFIX_RE.sub("", heading or "").strip()


def section_number(heading: str) -> str | None:
    """
    Leading section number of a heading, normalised to dotted arabic:
    '1.1 Definitions' → '1.1'; 'Clause 22.4' → '22.4'; 'ARTICLE IV …' → '4';
    'Schedule 2 …' → None (named sections are matched by prefix instead).
    """
    h = _base_heading(heading)
    label = _LEADING_LABEL_RE.match(h)
    had_label = bool(label)
    if label:
        h = h[label.end():].lstrip()

    m = re.match(r"(\d+(?:\.\d+)*)", h)
    if m:
        return m.group(1).rstrip(".")
    if had_label:
        rm = _ROMAN_RE.match(h)
        if rm and rm.group(1):
            value = _roman_to_int(rm.group(1))
            if 1 <= value <= _MAX_ROMAN_SECTION:
                return str(value)
    return None


def named_prefix(heading: str) -> tuple[str, str] | None:
    """'Schedule 2 Compounded Rate Terms' → ('schedule', '2')."""
    m = _NAMED_PREFIX_RE.match(_base_heading(heading))
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def _term_variants(term: str) -> list[str]:
    """The term plus its singular/plural counterpart ('Borrowers' ↔ 'Borrower')."""
    t = (term or "").strip()
    if not t:
        return []
    variants = [t]
    low = t.lower()
    if low.endswith("ies"):
        variants.append(t[:-3] + "y")
    elif low.endswith("s") and len(t) > 3:
        variants.append(t[:-1])
    else:
        variants.append(t + "s")
    return variants


def _definition_re(term: str) -> re.Pattern:
    """Matches '"Term" means / shall mean / has the meaning …'."""
    return re.compile(
        r"[\"“”']?" + re.escape(term.strip()) +
        r"[\"“”']?\s*(?:\([^)]{0,40}\)\s*)?(?:means|shall\s+mean|has\s+the\s+meaning|"
        r"is\s+defined|refers\s+to)",
        re.IGNORECASE,
    )


class LexicalIndex:
    """
    Deterministic, zero-cost lookups that neither dense nor sparse search can be
    trusted to get right:

      • definition_chunks("Margin")  → the chunk that DEFINES the term. A vector
        search for "Margin" returns every clause that USES it; the definition is
        what you actually need, and it is findable exactly.
      • section_chunks("Clause 22.4") → the cross-referenced section itself,
        including all of its '(part N)' splits.
      • phrase_chunks("...")         → verbatim substring hits, which is how you
        find a quoted covenant that was pasted into the input text.
    """

    def __init__(self, chunks: list[_ChunkView]):
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._by_section: dict[str, list[int]] = defaultdict(list)
        self._by_named: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            num = section_number(c.heading)
            if num:
                self._by_section[num].append(i)
            pref = named_prefix(c.heading)
            if pref:
                self._by_named[pref].append(i)

    # -- definitions ----------------------------------------------------------

    def definition_chunks(self, term: str) -> list[int]:
        """Indices of chunks that define `term` (or its singular/plural form)."""
        for variant in _term_variants(term):
            if len(variant) < 3:
                continue
            pattern = _definition_re(variant)
            hits = [i for i, c in enumerate(self.chunks) if pattern.search(c.full_text)]
            if hits:
                return hits
        return []

    def defines(self, term: str) -> bool:
        return bool(self.definition_chunks(term))

    # -- cross-references -----------------------------------------------------

    def section_chunks(self, ref: str) -> list[int]:
        """Indices of chunks belonging to a cross-referenced section."""
        num = section_number(ref)
        if num and num in self._by_section:
            return list(self._by_section[num])
        pref = named_prefix(ref)
        if pref and pref in self._by_named:
            return list(self._by_named[pref])
        return []

    def heading_chunks(self, term: str) -> list[int]:
        """Indices of chunks whose heading starts with `term` after its number."""
        t = _normalize(term)
        if len(t) < 3:
            return []
        out = []
        for i, c in enumerate(self.chunks):
            head = _normalize(_base_heading(c.heading)).lstrip("0123456789. ")
            if head.startswith(t):
                out.append(i)
        return out

    # -- verbatim phrases -----------------------------------------------------

    def phrase_chunks(self, phrase: str, min_len: int = 12) -> list[int]:
        """Indices of chunks containing `phrase` verbatim (encoding-normalised)."""
        needle = _normalize(phrase)
        if len(needle) < min_len:
            return []
        return [i for i, c in enumerate(self.chunks) if needle in c.norm_text]

    # -- structural neighbours ------------------------------------------------

    def sibling_parts(self, idx: int) -> list[int]:
        """Other '(part N)' splits of the same section — a clause cut in half is
        a classic source of a technically-retrieved-but-useless answer."""
        base = _normalize(_base_heading(self.chunks[idx].heading))
        if not base:
            return []
        return [i for i, c in enumerate(self.chunks)
                if i != idx and _normalize(_base_heading(c.heading)) == base]


# ── Reference extraction (defined terms & cross-refs inside a passage) ────────

_QUOTED_TERM_RE = re.compile(r'["“]([A-Z][A-Za-z0-9&\-\' ]{2,60}?)["”]')
_SECTION_REF_RE = re.compile(
    r"\b("
    r"(?:Section|Clause|Article|Paragraph)s?\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+\b)"
    r"|§\s*\d+(?:\.\d+)*"
    r"|(?:Schedule|Annex|Appendix|Exhibit|Part)\s+(?:\d+(?:\.\d+)*|[A-Z]\b|[IVXLCDM]+\b)"
    r")",
    re.IGNORECASE,
)
_CAP_PHRASE_RE = re.compile(
    r"\b((?:[A-Z][a-z][A-Za-z0-9\-']*|[A-Z]{2,})"
    r"(?:[ \t]+(?:[A-Z][a-z][A-Za-z0-9\-']*|[A-Z]{2,}))*)\b"
)
# Numbers that matter in a covenant: amounts, percentages, ratios, dates.
_VALUE_RE = re.compile(
    r"(?:[A-Z]{3}\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|m|bn))?"
    r"|[£$€]\s?[\d,]+(?:\.\d+)?(?:\s?(?:million|billion|m|bn))?"
    r"|\d+(?:\.\d+)?\s?(?:per\s?cent\.?|percent|%)"
    r"|\d+(?:\.\d+)?\s?:\s?\d+(?:\.\d+)?"
    r")",
    re.IGNORECASE,
)

_TERM_STOPWORDS = {
    "the", "this", "that", "these", "those", "a", "an", "if", "in", "on", "of",
    "no", "any", "each", "all", "such", "subject", "notwithstanding", "provided",
    "section", "clause", "article", "paragraph", "schedule", "annex", "appendix",
    "exhibit", "part", "page", "means", "shall", "may", "definitions", "and",
    "or", "but", "however", "further", "furthermore", "for", "to", "with",
}


def extract_references(text: str) -> tuple[list[str], list[str], list[str]]:
    """
    Deterministically pull (defined_terms, section_refs, values) from a passage.

    defined_terms : quoted terms first (certain), then capitalised phrases
                    (candidates — the caller should only follow the ones the
                    document actually defines, or prose noise gets chased).
    section_refs  : 'Clause 22.4', 'Schedule 2', '§ 5.1' …
    values        : amounts, percentages and ratios — high-signal sparse queries.
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = " ".join((t or "").split())
        n = _normalize(t)
        if len(n) < 3 or n in _TERM_STOPWORDS or n in seen:
            return
        seen.add(n)
        terms.append(t)

    for m in _QUOTED_TERM_RE.finditer(text or ""):
        _add(m.group(1))

    # Strip cross-refs first so "Clause" itself never surfaces as a term.
    stripped = _SECTION_REF_RE.sub(" ", text or "")
    for m in _CAP_PHRASE_RE.finditer(stripped):
        words = m.group(1).split()
        while words and words[0].lower() in _TERM_STOPWORDS:
            words = words[1:]
        while words and words[-1].lower() in _TERM_STOPWORDS:
            words = words[:-1]
        if words:
            _add(" ".join(words))

    refs: list[str] = []
    seen_refs: set[str] = set()
    for m in _SECTION_REF_RE.finditer(text or ""):
        ref = " ".join(m.group(1).split())
        key = _normalize(ref)
        if key not in seen_refs:
            seen_refs.add(key)
            refs.append(ref)

    values: list[str] = []
    seen_vals: set[str] = set()
    for m in _VALUE_RE.finditer(text or ""):
        val = " ".join(m.group(0).split())
        key = _normalize(val)
        if key not in seen_vals:
            seen_vals.add(key)
            values.append(val)

    return terms, refs, values


# ── Result models ─────────────────────────────────────────────────────────────

class Aspect(BaseModel):
    """
    One retrievable ITEM decomposed out of the input text.

    The unit of coverage: retrieval succeeds when every aspect has at least one
    chunk graded >= min_grade. `queries` is the full trail of search strings
    tried for this aspect, including the agent's refinements.
    """
    id: int
    text: str
    kind: str = "clause"          # "clause" | "term" | "section_ref"
    key_terms: list[str] = Field(default_factory=list)
    section_refs: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    covered: bool = False
    best_grade: int = 0
    evidence: list[int] = Field(default_factory=list)   # chunk_ids, best-first

    @property
    def label(self) -> str:
        return self.text if len(self.text) <= 80 else self.text[:77] + "…"


class RetrievedItem(BaseModel):
    """
    One retrieved chunk, with the provenance needed to audit WHY it is here.

    grade    : 0-3 relevance to its best-matching aspect (3 = directly states it).
    covers   : ids of every aspect this chunk is evidence for.
    sources  : retrieval channels that surfaced it ("dense", "sparse",
               "lexical:definition", "lexical:section", "hyde", …).
    is_support: True for items pulled in by definition-graph expansion rather
               than by matching an aspect directly.
    """
    chunk_id: int
    heading: str
    content: str
    pages: list = Field(default_factory=list)
    score: float = 0.0                 # fused retrieval score (RRF)
    grade: int = 0
    covers: list[int] = Field(default_factory=list)
    covers_text: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    is_support: bool = False
    via: str = ""                      # e.g. "aspect 3" | "term: Permitted Disposal"
    depth: int = 0                     # 0 = direct hit, 1+ = graph expansion
    rationale: str | None = None       # reranker's one-line justification

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()

    def __str__(self) -> str:
        tag = "support" if self.is_support else f"grade {self.grade}"
        return f"[{self.heading}] (pages {self.pages}; {tag})"


class RetrievalResult(BaseModel):
    """
    The retrieved set, plus the coverage bookkeeping that makes it auditable.

    Behaves like a list — `for item in result`, `len(result)`, `result[0]` —
    because the retrieved items are the point; the rest is diagnostics.
    """
    query: str
    items: list[RetrievedItem] = Field(default_factory=list)
    aspects: list[Aspect] = Field(default_factory=list)
    rounds_used: int = 0
    llm_used: bool = False
    dense_used: bool = False
    stats: dict = Field(default_factory=dict)

    # -- list-like behaviour ---------------------------------------------------

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    # -- convenience -----------------------------------------------------------

    @property
    def contents(self) -> list[str]:
        return [it.full_text for it in self.items]

    @property
    def primary(self) -> list[RetrievedItem]:
        """Items that directly cover an aspect (excludes graph-expansion supports)."""
        return [it for it in self.items if not it.is_support]

    @property
    def supports(self) -> list[RetrievedItem]:
        return [it for it in self.items if it.is_support]

    @property
    def covered_aspects(self) -> list[Aspect]:
        return [a for a in self.aspects if a.covered]

    @property
    def uncovered_aspects(self) -> list[Aspect]:
        return [a for a in self.aspects if not a.covered]

    @property
    def coverage(self) -> float:
        """Fraction of decomposed items with at least one confident chunk."""
        return (len(self.covered_aspects) / len(self.aspects)) if self.aspects else 0.0

    @property
    def combined_context(self) -> str:
        """Every retrieved item as one source-tagged block for the next LLM step."""
        parts = []
        for i, it in enumerate(self.items, 1):
            covers = f"; covers items {it.covers}" if it.covers else ""
            parts.append(
                f"[{i}] SECTION: {it.heading}  (pages {it.pages}{covers})\n"
                f"{'─' * 60}\n{it.full_text}"
            )
        return "\n\n".join(parts) if parts else "(nothing retrieved)"

    def items_for(self, aspect_id: int) -> list[RetrievedItem]:
        """Items covering a given aspect, best-first."""
        return [it for it in self.items if aspect_id in it.covers]

    def coverage_report(self) -> str:
        """Human-readable audit: what was found for each item, and what was not."""
        lines = [
            f"Coverage: {len(self.covered_aspects)}/{len(self.aspects)} item(s) "
            f"({self.coverage:.0%}) · {len(self.primary)} passage(s) "
            f"+ {len(self.supports)} support · {self.rounds_used} round(s)"
        ]
        for a in self.aspects:
            mark = "✓" if a.covered else "✗"
            heads = [it.heading for it in self.items_for(a.id)][:3]
            lines.append(f"  {mark} [{a.id}] {a.label}")
            if heads:
                lines.append(f"        → {'; '.join(h[:60] for h in heads)}")
            else:
                lines.append(f"        → nothing retrieved (tried: {a.queries[:4]})")
        return "\n".join(lines)


# ── LLM plumbing (optional, budgeted, always degradable) ──────────────────────

class LLM:
    """
    Thin, fault-tolerant wrapper around the chat endpoint.

    Every LLM stage in this module is an ENHANCEMENT, never a dependency: the
    first failure flips `available` off and all later stages silently take their
    heuristic path. A call budget stops a pathological input from fanning out
    into hundreds of requests.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        enabled: bool = True,
        max_calls: int = MAX_LLM_CALLS,
        verbose: bool = True,
    ):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.verbose = verbose
        self.max_calls = max_calls
        self.calls = 0
        self.available = bool(enabled and _LANGCHAIN_AVAILABLE)
        self._client = None
        if enabled and not _LANGCHAIN_AVAILABLE and verbose:
            print("      → langchain_openai unavailable; running fully heuristic")

    def _client_or_none(self):
        if self._client is None and self.available:
            try:
                self._client = ChatOpenAI(
                    model=self.model, base_url=self.base_url,
                    api_key=self.api_key, temperature=0,
                )
            except Exception as exc:                       # noqa: BLE001
                self._disable(exc)
        return self._client

    def _disable(self, exc: Exception) -> None:
        if self.available and self.verbose:
            print(f"      → LLM unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to heuristics for the rest of this run")
        self.available = False

    def chat(self, system: str, human: str) -> str:
        """Return the model's text, or "" if unavailable / over budget."""
        if not self.available or self.calls >= self.max_calls:
            return ""
        client = self._client_or_none()
        if client is None:
            return ""
        try:
            self.calls += 1
            out = client.invoke([
                SystemMessage(content=system),
                HumanMessage(content=human),
            ]).content
            return out if isinstance(out, str) else str(out or "")
        except Exception as exc:                            # noqa: BLE001
            self._disable(exc)
            return ""


def parse_json_array(raw: str) -> list:
    """Tolerantly pull a JSON array out of a model response."""
    if not raw:
        return []
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


# ── Stage 1: decompose the input into aspects ─────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z\"“(])")
_BULLET_RE = re.compile(r"^\s*(?:[-•*–]|\(?[a-z0-9]{1,3}[.)])\s+", re.IGNORECASE)

MIN_ASPECT_CHARS = 20       # below this a segment is a fragment, not an item
MIN_ASPECT_TOKENS = 3       # …as is anything with fewer content words than this
MAX_ASPECT_CHARS = 400


def heuristic_aspects(text: str, max_aspects: int = MAX_ASPECTS) -> list[str]:
    """
    Split the input into candidate items without an LLM.

    Bullets and numbered list entries are natural item boundaries; otherwise we
    split on sentence enders. Fragments (too short, or too few content words) are
    glued onto the previous segment — a bare "…and the Margin." is not an item —
    and very long ones are hard-split so a single run-on paragraph still yields
    more than one retrieval unit.

    Line breaks inside a paragraph are NOT boundaries — pasted text is usually
    hard-wrapped, and splitting on the wrap would cut sentences in half ("…a
    Leverage Ratio of not" / "more than 3.5:1…"), producing two half-items that
    each retrieve badly.
    """
    blocks = [b for b in re.split(r"\n\s*\n", text or "") if b.strip()]
    segments: list[str] = []

    for block in blocks:
        lines = block.split("\n")
        bulleted = [ln for ln in lines if _BULLET_RE.match(ln)]
        # A bulleted block: each bullet is its own item.
        if len(bulleted) >= 2:
            current = ""
            for ln in lines:
                if _BULLET_RE.match(ln):
                    if current.strip():
                        segments.append(current.strip())
                    current = _BULLET_RE.sub("", ln)
                else:
                    current += " " + ln
            if current.strip():
                segments.append(current.strip())
            continue
        # Unwrap the paragraph first, then split on sentence enders only.
        for part in _SENTENCE_SPLIT_RE.split(" ".join(block.split())):
            if part.strip():
                segments.append(part.strip())

    # Glue short fragments forward, hard-split over-long ones.
    merged: list[str] = []
    for seg in segments:
        seg = " ".join(seg.split())
        # Glue only true fragments forward. A short but complete sentence
        # ("Disposals need consent.") is its own item and must keep its own budget.
        if merged and (len(seg) < MIN_ASPECT_CHARS
                       or len(_tokenize(seg)) < MIN_ASPECT_TOKENS):
            merged[-1] = (merged[-1] + " " + seg).strip()
            continue
        if len(seg) > MAX_ASPECT_CHARS:
            words, cur = seg.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > MAX_ASPECT_CHARS:
                    merged.append(cur.strip())
                    cur = w
                else:
                    cur += " " + w
            if cur.strip():
                merged.append(cur.strip())
            continue
        merged.append(seg)

    # Drop list lead-ins ("Key terms to check:", "The Borrower must ensure that:")
    # and fragments with almost no content words. They are not retrievable items:
    # they would burn a retrieval budget and then be reported as "not found".
    out = []
    for s in merged:
        if len(s) < 12:
            continue
        if s.endswith(":") and len(s) < 80:
            continue
        if len(_tokenize(s)) < 3:
            continue
        out.append(s)
    return out[:max_aspects]


_DECOMPOSE_SYSTEM = (
    "You split a passage of legal/commercial text into the distinct ITEMS it "
    "mentions, so each item can be looked up separately in an agreement.\n\n"
    "Rules:\n"
    "- One item = one self-contained fact, obligation, definition, amount or "
    "requirement. If a sentence mentions two obligations, split it.\n"
    "- Keep the item's own wording; do NOT summarise into abstract labels.\n"
    "- Resolve pronouns so each item stands alone (\"it\" → the thing meant).\n"
    "- Do not invent items that are not in the text; do not merge distinct ones.\n\n"
    "Return ONLY a JSON array. Each element:\n"
    '  {"item": "<the self-contained item text>", '
    '"key_terms": ["<defined term or distinctive phrase to search for>", ...]}\n'
    "key_terms should be terms that would appear VERBATIM in the agreement "
    "(defined terms, clause labels, amounts, ratios), 0-4 per item."
)


def llm_aspects(text: str, llm: LLM, max_aspects: int = MAX_ASPECTS
                ) -> list[tuple[str, list[str]]]:
    """
    LLM decomposition of the input into (item, key_terms) pairs.

    Long inputs are mapped block-by-block so nothing is truncated away — the
    whole point is that items near the END of a long field must not be lost.
    Returns [] when the LLM is unavailable, and the caller falls back.
    """
    if not llm.available or not (text or "").strip():
        return []

    blocks, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(current) + len(para) > 6000 and current:
            blocks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current.strip():
        blocks.append(current)

    out: list[tuple[str, list[str]]] = []
    for block in blocks:
        data = parse_json_array(llm.chat(_DECOMPOSE_SYSTEM, block))
        for entry in data:
            if isinstance(entry, dict):
                item = str(entry.get("item", "") or "").strip()
                terms = entry.get("key_terms") or []
                terms = [str(t).strip() for t in terms if str(t).strip()][:4]
            else:
                item, terms = str(entry).strip(), []
            if len(item) >= 12:
                out.append((item, terms))
        if len(out) >= max_aspects:
            break
    return out[:max_aspects]


def build_aspects(
    text: str,
    llm: LLM,
    lexical: LexicalIndex,
    max_aspects: int = MAX_ASPECTS,
    verbose: bool = True,
) -> list[Aspect]:
    """
    Decompose the input into Aspects, enriched with the deterministic signals
    (defined terms, cross-references, amounts) found in each item's own text.

    Cross-references that appear anywhere in the input but landed in no aspect's
    reference list are promoted to their OWN aspects: if the text says "as
    permitted under Clause 22.4", that clause must be retrieved whether or not
    the decomposer thought it was a separate item.
    """
    pairs = llm_aspects(text, llm, max_aspects)
    source = "LLM"
    if not pairs:
        pairs = [(s, []) for s in heuristic_aspects(text, max_aspects)]
        source = "heuristic"
    # A short input ("Margin", "Clause 22.4") is one item, not zero — the
    # segmenters' minimum lengths must never swallow the whole query.
    if not pairs and text.strip():
        pairs = [(" ".join(text.split()), [])]
        source = "whole-input"

    aspects: list[Aspect] = []
    seen_tokens: list[set] = []
    for item, llm_terms in pairs:
        toks = set(_tokenize(item))
        # Drop near-duplicate items (the LLM sometimes restates one item twice).
        if any(_jaccard(toks, prev) > 0.85 for prev in seen_tokens):
            continue
        seen_tokens.append(toks)

        terms, refs, values = extract_references(item)
        # Keep only capitalised phrases the document actually defines or titles —
        # everything else is prose noise and would spend budget on nothing.
        kept = [t for t in terms if lexical.defines(t) or lexical.heading_chunks(t)]
        for t in llm_terms:
            if t and not any(_normalize(t) == _normalize(k) for k in kept):
                kept.append(t)

        aspects.append(Aspect(
            id=len(aspects), text=item, kind="clause",
            key_terms=kept[:5], section_refs=refs[:4], values=values[:4],
        ))
        if len(aspects) >= max_aspects:
            break

    # Promote un-assigned cross-references to first-class aspects.
    assigned = {_normalize(r) for a in aspects for r in a.section_refs}
    _, all_refs, _ = extract_references(text)
    for ref in all_refs:
        if len(aspects) >= max_aspects:
            break
        if _normalize(ref) in assigned:
            continue
        assigned.add(_normalize(ref))
        aspects.append(Aspect(
            id=len(aspects), text=f"The text of {ref}",
            kind="section_ref", section_refs=[ref],
        ))

    if verbose:
        print(f"      → {len(aspects)} item(s) decomposed ({source}); "
              f"e.g. {[a.label[:48] for a in aspects[:3]]}")
    return aspects


# ── Stage 2: query planning (per aspect) ──────────────────────────────────────

_HYDE_SYSTEM = (
    "You write ONE short passage of contract language (2-4 sentences) exactly as "
    "it would appear in a loan or commercial agreement, expressing the item given "
    "to you. Use formal drafting style and the defined-term capitalisation such an "
    "agreement would use. Output ONLY the clause text — no heading, no commentary."
)


def plan_queries(aspect: Aspect, llm: LLM, use_hyde: bool = True) -> list[str]:
    """
    Build the query set for one aspect.

    Four complementary angles, because each fails differently:
      • the item text itself           — dense semantic match
      • each key term                  — focused, beats the long item's average
      • '"Term" means'                 — the DEFINITION, not the usages
      • the values ("3.5:1", "10%")    — sparse exact match on identifiers
      • a HyDE clause (optional)       — bridges "description of a clause" and
        "the clause", which embed surprisingly far apart
    """
    queries: list[str] = []

    def _add(q: str) -> None:
        q = " ".join((q or "").split())
        if len(q) < 3:
            return
        if any(_normalize(q) == _normalize(existing) for existing in queries):
            return
        queries.append(q)

    _add(aspect.text[:400])
    for t in aspect.key_terms:
        _add(t)
        if t[:1].isupper():
            _add(f'"{t}" means')
    for r in aspect.section_refs:
        _add(r)
    for v in aspect.values:
        _add(v)

    if use_hyde and llm.available and aspect.kind == "clause":
        hyde = llm.chat(_HYDE_SYSTEM, aspect.text[:1200]).strip()
        if len(hyde) > 40:
            _add(hyde[:1200])

    return queries


_REFINE_SYSTEM = (
    "You are refining a document search that has FAILED to find an item in a "
    "legal agreement.\n\n"
    "You are given the item, the search queries already tried, and the headings "
    "of what those queries returned (which did not contain the item).\n\n"
    "Propose 3 NEW search queries that attack the item from genuinely DIFFERENT "
    "angles — a synonym drafters would use, the likely DEFINED TERM, the "
    "operative clause the item would live in, or the section heading it would sit "
    "under. Do not repeat or lightly reword a query already tried.\n\n"
    'Return ONLY a JSON array of strings: ["query one", "query two", "query three"]'
)


def refine_queries(aspect: Aspect, seen_headings: list[str], llm: LLM,
                   max_new: int = 3) -> list[str]:
    """
    The agentic step: ask the model for genuinely different angles on an item
    that retrieval missed, given what it has already seen.

    With no LLM, fall back to deterministic reformulations — the key terms on
    their own, and a definition probe — which is exactly what a human would try
    next when a long query returns nothing.
    """
    if llm.available:
        human = (
            f"ITEM:\n{aspect.text[:800]}\n\n"
            f"QUERIES ALREADY TRIED:\n" + "\n".join(f"- {q[:120]}" for q in aspect.queries[:8])
            + "\n\nHEADINGS RETURNED (none contained the item):\n"
            + ("\n".join(f"- {h[:100]}" for h in seen_headings[:8]) or "- (nothing)")
        )
        proposals = parse_json_array(llm.chat(_REFINE_SYSTEM, human))
        out = []
        for p in proposals:
            q = " ".join(str(p).split())
            if len(q) >= 3 and not any(_normalize(q) == _normalize(t) for t in aspect.queries):
                out.append(q)
        if out:
            return out[:max_new]

    # Heuristic fallback: individual key terms and a definition probe.
    fallback: list[str] = []
    for t in aspect.key_terms:
        for cand in (t, f'"{t}" means', f"{t} shall mean"):
            if not any(_normalize(cand) == _normalize(q) for q in aspect.queries):
                fallback.append(cand)
    # Content words alone, stripped of the item's connective bulk.
    core = " ".join(_tokenize(aspect.text)[:8])
    if core and not any(_normalize(core) == _normalize(q) for q in aspect.queries):
        fallback.append(core)
    return fallback[:max_new]


# ── Stage 3: hybrid retrieval + Reciprocal Rank Fusion ────────────────────────

def _dense_search(index, query: str, top_k: int) -> list[tuple[int, float]]:
    """
    Vector search against the index's store. Returns [(chunk_id, score)].
    Any failure (no store, endpoint down) degrades to no dense channel rather
    than taking the whole retrieval down.
    """
    store = _get(index, "vector_store", None)
    if store is None:
        return []
    try:
        hits = store.similarity_search_with_score(query, k=top_k)
    except Exception:                                       # noqa: BLE001
        return []
    best: dict[int, float] = {}
    for doc, score in hits:
        cid = (doc.metadata or {}).get("chunk_id")
        if cid is None:
            continue
        cid = int(cid)
        if cid not in best or score > best[cid]:
            best[cid] = float(score)
    return sorted(best.items(), key=lambda kv: -kv[1])


def _rrf_fuse(rankings: list[tuple[str, list[int]]], k: int = RRF_K
              ) -> tuple[dict[int, float], dict[int, set]]:
    """
    Reciprocal Rank Fusion over ranked lists from different channels.

        score(d) = Σ_channels weight_c / (k + rank_c(d))

    RRF is used instead of score normalisation because BM25 scores and cosine
    similarities live on incomparable scales, and any min-max normalisation is
    hostage to the single best hit in each list. Rank position is robust.

    Args:
        rankings: [(channel_label, [chunk_id ranked best-first]), ...]
    Returns:
        (fused_score_by_chunk_id, source_labels_by_chunk_id)
    """
    fused: dict[int, float] = defaultdict(float)
    sources: dict[int, set] = defaultdict(set)
    for label, ids in rankings:
        base = label.split(":")[0]
        weight = CHANNEL_WEIGHTS.get(base, 1.0)
        for rank, cid in enumerate(ids, start=1):
            fused[cid] += weight / (k + rank)
            sources[cid].add(label)
    return dict(fused), dict(sources)


def retrieve_candidates(
    index,
    chunks: list[_ChunkView],
    bm25: BM25Index,
    lexical: LexicalIndex,
    queries: list[str],
    top_k: int = TOP_K_PER_QUERY,
    dense: bool = True,
) -> tuple[dict[int, float], dict[int, set]]:
    """
    Run every query through every channel and fuse the rankings.

    Returns (fused_score_by_chunk_id, sources_by_chunk_id).
    """
    id_by_index = [c.chunk_id for c in chunks]
    rankings: list[tuple[str, list[int]]] = []

    for qi, q in enumerate(queries):
        if dense:
            hits = _dense_search(index, q, top_k)
            if hits:
                rankings.append((f"dense:q{qi}", [cid for cid, _ in hits]))

        sparse = bm25.search(q, top_k)
        if sparse:
            rankings.append((f"sparse:q{qi}", [id_by_index[i] for i, _ in sparse]))

        # Deterministic channels — only fire when the query looks like the thing
        # they resolve, so they stay high-precision.
        sec_hits = lexical.section_chunks(q)
        if sec_hits:
            rankings.append((f"lexical:section:q{qi}", [id_by_index[i] for i in sec_hits]))

        term = q[1:-1] if (q.startswith('"') and q.endswith('"')) else q
        term = re.sub(r'^"|"\s*(?:means|shall mean).*$', "", term).strip('" ')
        if 3 <= len(term) <= 60:
            def_hits = lexical.definition_chunks(term)
            if def_hits:
                rankings.append((f"lexical:definition:q{qi}",
                                 [id_by_index[i] for i in def_hits]))
            head_hits = lexical.heading_chunks(term)
            if head_hits:
                rankings.append((f"lexical:heading:q{qi}",
                                 [id_by_index[i] for i in head_hits[:top_k]]))

        phrase_hits = lexical.phrase_chunks(q)
        if phrase_hits:
            rankings.append((f"lexical:phrase:q{qi}",
                             [id_by_index[i] for i in phrase_hits[:top_k]]))

    return _rrf_fuse(rankings)


# ── Stage 4: MMR diversification ──────────────────────────────────────────────

def _chunk_vectors(index) -> dict[int, list[float]]:
    """
    Harvest the embedding vectors already stored in the vector store, keyed by
    chunk_id. Free — no embedding calls — and enables true semantic MMR. Returns
    {} for any store that does not expose them, and MMR falls back to lexical.
    """
    store = _get(index, "vector_store", None)
    raw = getattr(store, "store", None) if store is not None else None
    if not isinstance(raw, dict):
        return {}
    out: dict[int, list[float]] = {}
    for entry in raw.values():
        try:
            cid = (entry.get("metadata") or {}).get("chunk_id")
            vec = entry.get("vector")
            if cid is not None and vec:
                out[int(cid)] = list(vec)
        except (AttributeError, TypeError, ValueError):
            continue
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def mmr_rerank(
    candidate_ids: list[int],
    scores: dict[int, float],
    by_id: dict[int, _ChunkView],
    vectors: dict[int, list[float]],
    lambda_mult: float = MMR_LAMBDA,
    top_n: int | None = None,
) -> list[int]:
    """
    Maximal Marginal Relevance: greedily pick the candidate maximising
        λ · relevance − (1 − λ) · max similarity to what is already picked.

    In a legal agreement the same covenant is often restated across a definition,
    an operative clause and a schedule; without MMR one item's slots fill with
    three copies of the same text while another item goes uncovered. Similarity
    uses the stored embeddings when available, token overlap otherwise.
    """
    if not candidate_ids:
        return []
    top_n = top_n or len(candidate_ids)
    hi = max((scores.get(c, 0.0) for c in candidate_ids), default=0.0) or 1.0

    def sim(a: int, b: int) -> float:
        if a in vectors and b in vectors:
            return _cosine(vectors[a], vectors[b])
        ca, cb = by_id.get(a), by_id.get(b)
        if ca is None or cb is None:
            return 0.0
        return _jaccard(ca.token_set, cb.token_set)

    remaining = list(candidate_ids)
    selected: list[int] = [remaining.pop(0)]          # best-scoring seed
    while remaining and len(selected) < top_n:
        best, best_val = None, -1e9
        for cid in remaining:
            rel = scores.get(cid, 0.0) / hi
            redundancy = max((sim(cid, s) for s in selected), default=0.0)
            val = lambda_mult * rel - (1.0 - lambda_mult) * redundancy
            if val > best_val:
                best, best_val = cid, val
        selected.append(best)
        remaining.remove(best)
    return selected


# ── Stage 5: reranking (LLM cross-encoder stand-in + lexical fallback) ────────

_RERANK_SYSTEM = (
    "You grade how well each candidate passage from a legal agreement answers or "
    "contains a specific ITEM.\n\n"
    "Grades:\n"
    "  3 = the passage DIRECTLY states/contains the item (the operative wording, "
    "the definition, or the exact figure).\n"
    "  2 = the passage substantively addresses the item (needed to establish it, "
    "e.g. the clause it cross-refers to, or states part of it).\n"
    "  1 = related topic but does not establish the item.\n"
    "  0 = irrelevant.\n\n"
    "Judge only what the passage TEXT says — never assume unseen content. "
    "Multiple passages may score 3.\n\n"
    "Return ONLY a JSON array, one object per candidate you were shown:\n"
    '  {"id": <candidate id>, "grade": 0-3, "why": "<max 12 words>"}'
)


def _value_kind(value: str) -> str | None:
    """Classify a figure so like can be compared with like."""
    v = _normalize(value)
    if re.search(r"\d\s*:\s*\d", v):
        return "ratio"
    if re.search(r"(?:%|per\s?cent|percent)", v):
        return "percentage"
    if re.search(r"[a-z]{3}\s?[\d,]|[£$€¥]", v):
        return "amount"
    return None


_KIND_PRESENT_RE = {
    "ratio": re.compile(r"\d\s*:\s*\d"),
    "percentage": re.compile(r"\d\s*(?:%|per\s?cent|percent)"),
    "amount": re.compile(r"(?:[a-z]{3}\s?[\d,]+|[£$€¥]\s?[\d,]+)"),
}


def _lexical_grade(aspect: Aspect, chunk: _ChunkView) -> int:
    """
    Deterministic grader used when no LLM is available.

    Combines four cheap signals that correlate well with usefulness:
      • how much of the item's distinctive vocabulary the chunk contains,
      • whether the chunk carries the item's exact values ("3.5:1", "10%"),
      • whether the chunk DEFINES or is titled with one of the key terms,
      • whether the chunk states a figure of the SAME KIND as the item's.

    That last signal matters more than it looks: when the document disagrees with
    the query — the case this pipeline exists to find — the figures differ, so
    exact-value matching fails and raw overlap drops just below threshold. A CP
    item quoting "USD 250,000,000" and a clause stating "USD 200,000,000" are
    about the same term; grading that passage away would report the clause as
    missing instead of as a deviation.
    """
    a_tokens = set(_tokenize(aspect.text))
    if not a_tokens:
        return 0
    overlap = len(a_tokens & chunk.token_set) / len(a_tokens)

    value_hit = any(_normalize(v) in chunk.norm_text for v in aspect.values if len(v) > 2)
    term_hit = any(_normalize(t) in chunk.norm_text for t in aspect.key_terms if len(t) > 2)
    ref_hit = any(section_number(r) and section_number(r) == section_number(chunk.heading)
                  for r in aspect.section_refs)
    kinds = {k for k in (_value_kind(v) for v in aspect.values) if k}
    kind_hit = any(_KIND_PRESENT_RE[k].search(chunk.norm_text) for k in kinds)

    if ref_hit or (value_hit and overlap >= 0.25) or overlap >= 0.6:
        return 3
    if (term_hit and overlap >= 0.2) or overlap >= 0.4 or (kind_hit and overlap >= 0.3):
        return 2
    if overlap >= 0.15 or term_hit or kind_hit:
        return 1
    return 0


def rerank_candidates(
    aspect: Aspect,
    candidate_ids: list[int],
    by_id: dict[int, _ChunkView],
    llm: LLM,
    snippet_chars: int = 900,
) -> dict[int, tuple[int, str | None]]:
    """
    Grade candidates 0-3 against the aspect. One batched LLM call per aspect
    (a pointwise cross-encoder stand-in); the lexical grader fills in for any
    candidate the model omitted, and for the whole set when there is no LLM.

    Returns {chunk_id: (grade, rationale)}.
    """
    graded: dict[int, tuple[int, str | None]] = {}
    if not candidate_ids:
        return graded

    if llm.available:
        lines = []
        for cid in candidate_ids:
            ch = by_id.get(cid)
            if ch is None:
                continue
            body = ch.content.strip()[:snippet_chars] or ch.heading
            lines.append(f"### CANDIDATE {cid}\nHEADING: {ch.heading}\nTEXT: {body}")
        human = f"ITEM:\n{aspect.text[:900]}\n\n" + "\n\n".join(lines)
        for entry in parse_json_array(llm.chat(_RERANK_SYSTEM, human)):
            if not isinstance(entry, dict):
                continue
            try:
                cid = int(entry.get("id"))
                grade = int(entry.get("grade", 0))
            except (TypeError, ValueError):
                continue
            if cid in by_id:
                why = entry.get("why")
                graded[cid] = (max(0, min(3, grade)), str(why) if why else None)

    for cid in candidate_ids:
        if cid not in graded and cid in by_id:
            graded[cid] = (_lexical_grade(aspect, by_id[cid]), None)
    return graded


# ── Stage 6: definition-graph expansion ───────────────────────────────────────

def expand_context(
    selected_ids: list[int],
    by_id: dict[int, _ChunkView],
    chunks: list[_ChunkView],
    lexical: LexicalIndex,
    max_items: int = MAX_SUPPORT_ITEMS,
    depth: int = EXPAND_DEPTH,
) -> list[tuple[int, str, int]]:
    """
    Breadth-first walk of the definition graph from the selected passages.

    A retrieved clause is rarely self-contained: "the Borrower may not make a
    Restricted Payment except a Permitted Distribution (see Clause 22.4)" is
    unusable without the two definitions and that clause. This follows quoted
    defined terms and cross-references out of the selected text, `depth` hops
    deep, and returns them as support items.

    Returns [(chunk_id, via, depth)] in discovery order.
    """
    out: list[tuple[int, str, int]] = []
    seen_chunks = set(selected_ids)
    seen_refs: set[str] = set()
    frontier = [by_id[c] for c in selected_ids if c in by_id]

    for d in range(1, depth + 1):
        next_frontier: list[_ChunkView] = []
        for ch in frontier:
            if len(out) >= max_items:
                break
            terms, refs, _ = extract_references(ch.content)

            wanted: list[tuple[str, str]] = []
            for r in refs:
                if _normalize(r) not in seen_refs:
                    seen_refs.add(_normalize(r))
                    wanted.append(("section", r))
            for t in terms:
                key = _normalize(t)
                if key in seen_refs:
                    continue
                seen_refs.add(key)
                # Only chase capitalised phrases the document really defines.
                if lexical.defines(t):
                    wanted.append(("term", t))

            for kind, value in wanted:
                if len(out) >= max_items:
                    break
                idxs = (lexical.section_chunks(value) if kind == "section"
                        else lexical.definition_chunks(value))
                for i in idxs[:2]:
                    cid = chunks[i].chunk_id
                    if cid in seen_chunks:
                        continue
                    seen_chunks.add(cid)
                    out.append((cid, f"{kind}: {value}", d))
                    next_frontier.append(chunks[i])
        if len(out) >= max_items or not next_frontier:
            break
        frontier = next_frontier
    return out[:max_items]


# ── Stage 7: coverage-first selection ─────────────────────────────────────────

def select_items(
    aspects: list[Aspect],
    graded_by_aspect: dict[int, list[tuple[int, int, float, str | None]]],
    by_id: dict[int, _ChunkView],
    sources: dict[int, set],
    max_items: int,
    min_grade: int,
) -> list[RetrievedItem]:
    """
    Pick the final set ROUND-ROBIN across aspects rather than by global score.

    This is what makes the output cover a multi-item input. A global top-k ranks
    every candidate on one scale, so the three passages about the item with the
    richest vocabulary crowd out the only passage about a tersely-worded one.
    Round-robin gives every item its first slot before any item gets its second.

    Pass 1 places each aspect's best confident (grade >= min_grade) evidence.
    Later passes distribute remaining budget, and a final pass admits weak
    (grade 1) evidence only for aspects that still have nothing.

    graded_by_aspect: {aspect_id: [(chunk_id, grade, fused_score, why), ...]}
    """
    chosen: dict[int, RetrievedItem] = {}
    order: list[int] = []
    queues = {a.id: list(graded_by_aspect.get(a.id, [])) for a in aspects}

    def _place(aspect: Aspect, cid: int, grade: int, score: float, why: str | None) -> bool:
        """Add or annotate an item; returns True if a new slot was consumed."""
        if cid in chosen:
            item = chosen[cid]
            if aspect.id not in item.covers:
                item.covers.append(aspect.id)
                item.covers_text.append(aspect.label)
                item.grade = max(item.grade, grade)
            return False
        ch = by_id.get(cid)
        if ch is None:
            return False
        chosen[cid] = RetrievedItem(
            chunk_id=cid, heading=ch.heading, content=ch.content, pages=ch.pages,
            score=round(score, 6), grade=grade, covers=[aspect.id],
            covers_text=[aspect.label], sources=sorted(sources.get(cid, set())),
            via=f"item {aspect.id}", rationale=why,
        )
        order.append(cid)
        return True

    def _drain(threshold: int, per_round: int = 1) -> None:
        """One round-robin sweep: up to `per_round` items per aspect at `threshold`."""
        progress = True
        while progress and len(order) < max_items:
            progress = False
            for a in aspects:
                if len(order) >= max_items:
                    return
                taken = 0
                queue = queues[a.id]
                while queue and taken < per_round:
                    cid, grade, score, why = queue[0]
                    if grade < threshold:
                        break                      # queue is grade-sorted; done here
                    queue.pop(0)
                    if _place(a, cid, grade, score, why):
                        taken += 1
                        progress = True
                        break
                    progress = True                # annotated an existing item
        return

    _drain(threshold=max(min_grade, 3))            # pass 1: only the direct hits
    _drain(threshold=min_grade)                    # pass 2: confident evidence
    # Pass 3: aspects still empty may take weak evidence rather than nothing.
    starved = [a for a in aspects if not any(a.id in it.covers for it in chosen.values())]
    if starved and len(order) < max_items:
        for a in starved:
            for cid, grade, score, why in queues[a.id]:
                if grade >= 1 and len(order) < max_items:
                    _place(a, cid, grade, score, why)
                    break

    items = [chosen[cid] for cid in order]
    items.sort(key=lambda it: (-it.grade, -it.score))
    return items


# ── The retriever ─────────────────────────────────────────────────────────────

class Retriever:
    """
    Reusable retriever over one ingested document.

    Building the sparse index and the lexical maps costs a pass over the chunks;
    constructing this once and calling `.retrieve()` many times avoids paying it
    per query. `retrieve()` (module-level) is the one-shot convenience wrapper.

        r = Retriever(index)
        result = r.retrieve(long_text)
    """

    def __init__(
        self,
        index,
        llm_model: str = DEFAULT_LLM_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = DEFAULT_API_KEY,
        use_llm: bool = True,
        max_llm_calls: int = MAX_LLM_CALLS,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        heading_weight: float = 3.0,
        verbose: bool = True,
    ):
        self.index = index
        self.verbose = verbose
        self.chunks = _adapt_chunks(index)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25 = BM25Index(self.chunks, k1=bm25_k1, b=bm25_b,
                              heading_weight=heading_weight)
        self.lexical = LexicalIndex(self.chunks)
        self.vectors = _chunk_vectors(index)
        self.has_dense = _get(index, "vector_store", None) is not None
        self._llm_config = dict(model=llm_model, base_url=base_url, api_key=api_key,
                                enabled=use_llm, max_calls=max_llm_calls,
                                verbose=verbose)
        if verbose:
            print(f"[retrieval] Indexed {len(self.chunks)} chunk(s) · "
                  f"sparse=BM25F · dense={'on' if self.has_dense else 'off'} · "
                  f"vectors cached={len(self.vectors)}")

    # -- main entry point -----------------------------------------------------

    def retrieve(
        self,
        text: str,
        max_items: int = MAX_ITEMS,
        max_aspects: int = MAX_ASPECTS,
        top_k_per_query: int = TOP_K_PER_QUERY,
        candidates_per_aspect: int = CANDIDATES_PER_ASPECT,
        max_rounds: int = MAX_ROUNDS,
        min_grade: int = MIN_GRADE,
        use_hyde: bool = True,
        use_mmr: bool = True,
        mmr_lambda: float = MMR_LAMBDA,
        expand_depth: int = EXPAND_DEPTH,
        max_support_items: int = MAX_SUPPORT_ITEMS,
        include_sibling_parts: bool = True,
        verbose: bool | None = None,
    ) -> RetrievalResult:
        """
        Retrieve every passage needed to cover `text`.

        Args:
            text:                  the free-text input; may be long and may talk
                                   about several unrelated parts of the document.
            max_items:             cap on aspect-covering passages returned.
            max_aspects:           cap on items decomposed out of the input.
            top_k_per_query:       candidates pulled per query, per channel.
            candidates_per_aspect: candidates that survive fusion into reranking.
            max_rounds:            retrieve→grade→refine rounds (1 disables the
                                   agentic refinement loop).
            min_grade:             grade (0-3) at which an item counts as covered.
            use_hyde:              generate a hypothetical clause per item to use
                                   as an extra dense query.
            use_mmr:               diversify each item's candidates before selection.
            expand_depth:          definition-graph hops from the selected items
                                   (0 disables support retrieval).
            max_support_items:     cap on definition/cross-reference items.
            include_sibling_parts: pull in '(part N)' splits of selected chunks so
                                   a clause is never returned cut in half.

        Returns:
            RetrievalResult — iterable as a list of RetrievedItem, plus
            `.aspects`, `.coverage`, `.uncovered_aspects` and `.coverage_report()`.
        """
        verbose = self.verbose if verbose is None else verbose
        llm = LLM(**self._llm_config)
        llm.verbose = verbose
        text = text or ""

        if verbose:
            print(f"\n[retrieval] Input: {len(text)} chars · "
                  f"budget {max_items} item(s) · rounds<={max_rounds}")

        # ── 1. decompose ──────────────────────────────────────────────────────
        aspects = build_aspects(text, llm, self.lexical, max_aspects, verbose)
        if not aspects:
            return RetrievalResult(query=text, llm_used=llm.calls > 0,
                                   dense_used=self.has_dense)

        # Per aspect: fused candidate scores and grades, accumulated over rounds.
        fused_by_aspect: dict[int, dict[int, float]] = {a.id: {} for a in aspects}
        graded_by_aspect: dict[int, dict[int, tuple[int, str | None]]] = {a.id: {} for a in aspects}
        sources: dict[int, set] = defaultdict(set)
        rounds_used = 0

        pending = list(aspects)
        for rnd in range(1, max_rounds + 1):
            if not pending:
                break
            rounds_used = rnd
            if verbose:
                print(f"      → round {rnd}: retrieving for {len(pending)} item(s)")

            issued_any = False
            for aspect in pending:
                # ── 2. plan queries ──────────────────────────────────────────
                if rnd == 1:
                    queries = plan_queries(aspect, llm, use_hyde=use_hyde)
                else:
                    seen_headings = [
                        self.by_id[c].heading for c in
                        sorted(fused_by_aspect[aspect.id],
                               key=lambda c: -fused_by_aspect[aspect.id][c])[:8]
                        if c in self.by_id
                    ]
                    queries = refine_queries(aspect, seen_headings, llm)
                    if not queries:
                        continue
                aspect.queries.extend(queries)
                issued_any = True

                # ── 3. hybrid retrieve + fuse ────────────────────────────────
                fused, srcs = retrieve_candidates(
                    self.index, self.chunks, self.bm25, self.lexical,
                    queries, top_k=top_k_per_query, dense=self.has_dense,
                )
                for cid, sc in fused.items():
                    fused_by_aspect[aspect.id][cid] = fused_by_aspect[aspect.id].get(cid, 0.0) + sc
                for cid, labels in srcs.items():
                    sources[cid].update(labels)

                ranked = sorted(fused_by_aspect[aspect.id].items(), key=lambda kv: -kv[1])
                candidate_ids = [cid for cid, _ in ranked][: candidates_per_aspect * 2]

                # ── 4. diversify ─────────────────────────────────────────────
                if use_mmr and len(candidate_ids) > 2:
                    candidate_ids = mmr_rerank(
                        candidate_ids, fused_by_aspect[aspect.id], self.by_id,
                        self.vectors, lambda_mult=mmr_lambda,
                        top_n=candidates_per_aspect,
                    )
                else:
                    candidate_ids = candidate_ids[:candidates_per_aspect]

                # ── 5. rerank (skip anything already graded this run) ─────────
                fresh = [c for c in candidate_ids if c not in graded_by_aspect[aspect.id]]
                if fresh:
                    graded_by_aspect[aspect.id].update(
                        rerank_candidates(aspect, fresh, self.by_id, llm)
                    )

                grades = graded_by_aspect[aspect.id]
                aspect.best_grade = max((g for g, _ in grades.values()), default=0)
                aspect.covered = aspect.best_grade >= min_grade

            # ── 6. coverage check → who goes into the next (agentic) round ────
            pending = [a for a in pending if not a.covered]
            if verbose:
                covered = sum(1 for a in aspects if a.covered)
                print(f"      → after round {rnd}: {covered}/{len(aspects)} item(s) "
                      f"covered" + (f"; refining {len(pending)}" if pending and rnd < max_rounds else ""))
            if not pending:
                break
            # Nothing new to try — another round would re-run identical queries.
            if not issued_any:
                if verbose:
                    print("      → no new search angles available; stopping early")
                break

        # ── 7. coverage-first selection ───────────────────────────────────────
        ranked_by_aspect: dict[int, list[tuple[int, int, float, str | None]]] = {}
        for a in aspects:
            rows = [
                (cid, grade, fused_by_aspect[a.id].get(cid, 0.0), why)
                for cid, (grade, why) in graded_by_aspect[a.id].items()
            ]
            rows.sort(key=lambda r: (-r[1], -r[2]))
            ranked_by_aspect[a.id] = rows
            a.evidence = [cid for cid, grade, _, _ in rows if grade >= min_grade]

        items = select_items(aspects, ranked_by_aspect, self.by_id, sources,
                             max_items=max_items, min_grade=min_grade)

        # ── 8. structural completion: sibling '(part N)' splits ───────────────
        selected_ids = [it.chunk_id for it in items]
        if include_sibling_parts:
            index_by_id = {c.chunk_id: i for i, c in enumerate(self.chunks)}
            extra: list[RetrievedItem] = []
            for it in items:
                i = index_by_id.get(it.chunk_id)
                if i is None:
                    continue
                for j in self.lexical.sibling_parts(i):
                    cid = self.chunks[j].chunk_id
                    if cid in selected_ids or any(e.chunk_id == cid for e in extra):
                        continue
                    ch = self.chunks[j]
                    extra.append(RetrievedItem(
                        chunk_id=cid, heading=ch.heading, content=ch.content,
                        pages=ch.pages, grade=0, is_support=True, depth=1,
                        via=f"sibling part of {_base_heading(it.heading)[:50]}",
                        sources=["structural"],
                    ))
            items.extend(extra[:max_support_items])
            selected_ids = [it.chunk_id for it in items]

        # ── 9. definition-graph expansion ─────────────────────────────────────
        if expand_depth > 0 and max_support_items > 0:
            budget = max(0, max_support_items - sum(1 for it in items if it.is_support))
            for cid, via, depth in expand_context(
                selected_ids, self.by_id, self.chunks, self.lexical,
                max_items=budget, depth=expand_depth,
            ):
                ch = self.by_id.get(cid)
                if ch is None:
                    continue
                items.append(RetrievedItem(
                    chunk_id=cid, heading=ch.heading, content=ch.content,
                    pages=ch.pages, grade=0, is_support=True, depth=depth,
                    via=via, sources=["graph"],
                ))

        result = RetrievalResult(
            query=text, items=items, aspects=aspects, rounds_used=rounds_used,
            llm_used=llm.calls > 0, dense_used=self.has_dense,
            stats={
                "chunks_indexed": len(self.chunks),
                "llm_calls": llm.calls,
                "queries_issued": sum(len(a.queries) for a in aspects),
                "candidates_graded": sum(len(g) for g in graded_by_aspect.values()),
                "primary_items": sum(1 for it in items if not it.is_support),
                "support_items": sum(1 for it in items if it.is_support),
            },
        )
        if verbose:
            print(f"      → {len(result.primary)} passage(s) + {len(result.supports)} "
                  f"support · coverage {result.coverage:.0%} · "
                  f"{llm.calls} LLM call(s)")
            if result.uncovered_aspects:
                print(f"      → NOT found: "
                      f"{[a.label[:60] for a in result.uncovered_aspects]}")
        return result


# ── Module-level convenience API ──────────────────────────────────────────────

def retrieve(index, text: str, **kwargs) -> RetrievalResult:
    """
    Retrieve every passage that covers `text`, one shot.

    Splits kwargs between the Retriever constructor (llm_model, base_url,
    api_key, use_llm, verbose, …) and the retrieve() call, so callers can pass
    anything from either without caring which is which.

        result = retrieve(index, long_text, max_items=30, use_hyde=False)
        for item in result:
            print(item.heading, item.pages, item.covers)

    Building a Retriever once (see the class) is cheaper across many queries.
    """
    ctor_keys = {"llm_model", "base_url", "api_key", "use_llm", "max_llm_calls",
                 "bm25_k1", "bm25_b", "heading_weight"}
    ctor = {k: v for k, v in kwargs.items() if k in ctor_keys}
    if "verbose" in kwargs:
        ctor["verbose"] = kwargs["verbose"]
    call = {k: v for k, v in kwargs.items() if k not in ctor_keys}
    return Retriever(index, **ctor).retrieve(text, **call)


def retrieve_items(index, text: str, **kwargs) -> list[RetrievedItem]:
    """`retrieve()` returning the plain list of items."""
    return retrieve(index, text, **kwargs).items


def retrieve_texts(index, text: str, **kwargs) -> list[str]:
    """`retrieve()` returning just the passage texts, best-first."""
    return [it.full_text for it in retrieve(index, text, **kwargs).items]


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.retrieval <pdf_path> <text|@file.txt>")
        raise SystemExit(1)

    from .ingestion_v3 import DocumentIndexV3   # demo only — not a module dependency

    arg = sys.argv[2]
    query_text = open(arg[1:], encoding="utf-8").read() if arg.startswith("@") else arg

    idx = DocumentIndexV3.from_pdf(sys.argv[1])
    res = Retriever(idx).retrieve(query_text)

    print("\n" + "=" * 70)
    print(res.coverage_report())
    print("=" * 70)
    for n, item in enumerate(res, 1):
        kind = "SUPPORT" if item.is_support else f"grade {item.grade}"
        print(f"\n[{n}] {item.heading}  (pages {item.pages}; {kind}; "
              f"score={item.score:.4f}; via {item.via})")
        if item.rationale:
            print(f"    why: {item.rationale}")
        print(f"    sources: {item.sources[:4]}")
        print("    " + item.content.strip()[:300].replace("\n", "\n    "))
    print(f"\nSTATS: {res.stats}")
