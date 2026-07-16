"""
Query mode: answer an open natural-language QUESTION about the document.

Main entry point: run_query(index, question, ...)

Why this is a different methodology from run_find
-------------------------------------------------
run_find (find_content.py) locates the passage that matches a *description* of
content — the query already reads like the answer's text. A *question* does not:
"What is the base currency?" embeds poorly against '"Base Currency" means Dollars
($)'. So a single similarity search routinely misses the answer.

run_query therefore uses an AGENTIC retrieve-refine loop (the same tool-calling
agent pattern as evidence.py): the LLM is handed a `search_document` tool and is
expected to issue its OWN search queries, read the passages, and RE-QUERY with
refined terms ("base currency" → "agreed currency" → "currency of the loan")
until it can answer or it gives up. It returns two things, not one:

  • answer    — the extracted value, e.g. "USD"
  • reference — the verbatim phrase from the document that proves it

Grounding guard (anti-hallucination)
-------------------------------------
Whatever the agent answers, each returned `reference` is checked DETERMINISTICALLY:
it must be a normalised verbatim substring of a document passage. Matching first
tries the retrieved pool (so references keep a similarity score) and then falls
back to the WHOLE document, so a genuine quote from a passage the agent never
retrieved still grounds. Normalisation (see _normalize) forgives only cosmetic
encoding differences — smart quotes, dashes, ligatures, whitespace, casing — never
paraphrase. If nothing grounds, found=False. This mirrors evidence.py's "the agent
decides, the code enforces truth" guarantee and extract.py's rule that the LLM
never invents document text.
"""

import json
import re
import unicodedata

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import ContentMatch, QueryReference, QueryResult
from .find_content import (
    materialize_candidate,
    retrieve_content_candidates,
    _to_match,
)
from .section_utils import chunk_to_doc


# ── Grounding (deterministic verbatim check) ──────────────────────────────────

# Punctuation variants that are visually/semantically identical but differ in
# code point — the commonest cause of a real quote failing to ground. Mapped to a
# single canonical form so both the quote and the source text agree. Applied to
# BOTH sides, so this only makes genuinely-identical text match; it can never turn
# a paraphrase into a match.
_PUNCT_MAP = {
    # single quotes / apostrophes → '
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "`": "'", "´": "'",
    # double quotes → "
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": '"',
    "«": '"', "»": '"',
    # dashes / minus → -
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "―": "-", "−": "-",
    # zero-width / soft hyphen / BOM → removed
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "",
    # ellipsis → three dots (so "…" and "..." agree)
    "…": "...",
}
_PUNCT_TABLE = str.maketrans(_PUNCT_MAP)


def _normalize(text: str) -> str:
    """
    Canonicalise text for grounding comparisons: NFKC-normalise (folds ligatures,
    full-width forms, non-breaking spaces, etc.), unify punctuation variants
    (smart quotes, en/em dashes, soft hyphens, ellipsis), collapse whitespace and
    lowercase. Applied identically to both the quote and the source, so a match
    still means the text is genuinely the same — only cosmetic encoding
    differences are forgiven.
    """
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _ground_reference(
    reference: str,
    pool: dict[int, tuple[Document, float]],
    whole_doc: list[tuple[Document, str]] | None = None,
) -> tuple[Document, float | None] | None:
    """
    Return the (doc, score) whose text verbatim-contains `reference` (after
    normalisation), or None if the quote is not grounded anywhere.

    We only accept an exact normalised substring, so a hallucinated/paraphrased
    quote fails — but normalisation (see _normalize) forgives cosmetic encoding
    differences (smart quotes, dashes, whitespace, casing, ligatures).

    Grounding is checked in two tiers:
      1. `pool` — the passages the agent's searches actually retrieved. A hit here
         carries its similarity score (preferred, so references keep a score).
      2. `whole_doc` — every chunk of the document, precomputed as
         (Document, normalised_text). This catches a genuine quote that lives in a
         passage the agent never happened to retrieve. A whole-doc-only hit has no
         similarity score (score=None); it is still a verbatim match.
    """
    needle = _normalize(reference)
    if len(needle) < 4:        # too short to be a meaningful, unique proof
        return None

    best: tuple[Document, float] | None = None
    for doc, score in pool.values():
        if needle in _normalize(doc.page_content):
            if best is None or score > best[1]:
                best = (doc, score)
    if best is not None:
        return best

    # Tier 2: whole-document fallback (normalised text precomputed by caller).
    if whole_doc:
        for doc, norm_text in whole_doc:
            if needle in norm_text:
                return (doc, None)
    return None


# ── Query-term extraction (coverage for long queries) ─────────────────────────
#
# A long question ("In computing the leverage ratio, how is Adjusted EBITDA
# defined and what add-backs are permitted?") embeds as one vector dominated by
# its bulk, so a nested key term like "Adjusted EBITDA" may not score high enough
# for its defining chunk to be retrieved at all — and a chunk that is never
# retrieved can never be cited. We extract the key terms and run a FOCUSED search
# per term, seeding the pool with those high-score chunks. A deterministic
# heuristic catches capitalised/acronym terms for free; the LLM extractor
# (reused from collate) adds semantic terms the heuristic misses.

# Question words and connectives that are never useful as standalone search terms.
_QUESTION_STOPWORDS = {
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "the", "a", "an", "of",
    "in", "on", "for", "to", "and", "or", "does", "do", "did", "will", "shall",
    "can", "could", "would", "should", "this", "that", "these", "those",
    "under", "per", "by", "with", "as", "at", "from", "any", "used",
}

# Runs of capitalised words / ALL-CAPS acronyms — e.g. "Adjusted EBITDA",
# "Final Maturity Date", "SONIA". Every token in the run must be capitalised or an
# acronym; lowercase connectives break the run (so "SONIA for GBP" → "SONIA",
# "GBP", not one glued term).
_CAP_RUN_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&/.-]+|[A-Z]{2,})"
    r"(?:\s+(?:[A-Z][A-Za-z0-9&/.-]+|[A-Z]{2,}))*"
)
_QUOTED_RE = re.compile(r"[\"“'‘]([^\"”'’]{2,60})[\"”'’]")


def _heuristic_key_terms(question: str) -> list[str]:
    """
    Deterministically pull nested key terms from the question: quoted phrases,
    multi-word capitalised phrases, and ALL-CAPS acronyms. Leading/trailing
    question stopwords are trimmed (so "What" in "What is Adjusted EBITDA" is
    dropped). Single capitalised common words are NOT kept — those are left to the
    LLM extractor — to avoid seeding noise.
    """
    terms: list[str] = []

    for m in _QUOTED_RE.finditer(question):
        phrase = m.group(1).strip()
        if phrase:
            terms.append(phrase)

    for m in _CAP_RUN_RE.finditer(question):
        words = m.group(0).split()
        while words and words[0].lower() in _QUESTION_STOPWORDS:
            words = words[1:]
        while words and words[-1].lower() in _QUESTION_STOPWORDS:
            words = words[:-1]
        if not words:
            continue
        phrase = " ".join(words)
        is_acronym = phrase.isupper() and len(phrase) >= 2
        if len(words) >= 2 or is_acronym:      # keep phrases & acronyms, not lone words
            terms.append(phrase)

    return terms


def _dedupe_preserve(terms: list[str], cap: int) -> list[str]:
    """Case-insensitively dedupe, drop stopword-only/short terms, keep order, cap."""
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        t = t.strip()
        key = _normalize(t)
        if len(key) < 3 or key in _QUESTION_STOPWORDS or key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= cap:
            break
    return out


def extract_query_terms(
    question: str,
    use_llm: bool = True,
    max_terms: int = 6,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[str]:
    """
    Extract key terms from a question to seed focused retrieval.

    Combines the deterministic heuristic (capitalised/acronym/quoted phrases,
    listed FIRST so nested terms like "Adjusted EBITDA" are prioritised) with the
    LLM extractor reused from collate (semantic terms the heuristic misses). The
    LLM step degrades gracefully — if it errors, the heuristic terms still stand.
    """
    terms = _heuristic_key_terms(question)

    if use_llm:
        try:
            from .collate import extract_key_terms
            terms = terms + extract_key_terms(
                question, llm_model=llm_model, base_url=base_url, api_key=api_key,
            )
        except Exception as exc:   # noqa: BLE001 — LLM/backend unavailable, etc.
            print(f"      → LLM term extraction unavailable "
                  f"({type(exc).__name__}); using heuristic terms only")

    # Never seed with the whole question as a "term"; it defeats the purpose.
    q_norm = _normalize(question)
    terms = [t for t in terms if _normalize(t) != q_norm]
    return _dedupe_preserve(terms, max_terms)


# ── Search tool (what the agent drives) ───────────────────────────────────────

def _make_search_tool(index: DocumentIndex, pool: dict[int, tuple[Document, float]],
                      search_terms: list[str], top_k: int):
    """
    Build the `search_document` StructuredTool. Every passage the tool surfaces
    is recorded in `pool` (keyed by chunk_id) so the grounding guard can later
    verify quotes against exactly what the agent was shown. The agent's query
    strings are recorded in `search_terms` as the refinement trail.
    """

    def search_document(query: str) -> str:
        """Search the document for passages relevant to `query` and return the
        top matches (heading + text). Call this repeatedly with REFINED queries
        — synonyms, defined terms, or rephrasings — until you find the answer."""
        if any(_normalize(query) == _normalize(t) for t in search_terms):
            return ("You already searched for that exact wording. Try a DIFFERENT "
                    "angle: a synonym, the likely defined term, or a rephrasing.")
        search_terms.append(query)
        hits = []
        for doc, score in retrieve_content_candidates(index, query, top_k=top_k):
            usable = materialize_candidate(doc, index)
            if usable is None:
                continue
            cid = usable.metadata.get("chunk_id")
            if cid is not None and (cid not in pool or score > pool[cid][1]):
                pool[cid] = (usable, score)
            hits.append(usable)

        if not hits:
            return "No passages found for that query. Try different wording."
        lines = []
        for d in hits:
            heading = d.metadata.get("heading", "(no heading)")
            body = d.page_content.strip()[:1200]
            lines.append(f"[{heading}]  (pages {d.metadata.get('pages', [])})\n{body}")
        return "\n\n---\n\n".join(lines)

    return StructuredTool.from_function(
        func=search_document,
        name="search_document",
        description=(
            "Search the legal document for passages relevant to a query string. "
            "Returns the most similar passages (heading + text). Call it multiple "
            "times with refined/alternative wording to track down the answer."
        ),
    )


# ── Agent prompt / output parsing ─────────────────────────────────────────────

_SYSTEM = (
    "You are a legal document analyst answering a precise question about ONE "
    "agreement. You cannot see the document directly — you must use the "
    "`search_document` tool to retrieve passages.\n\n"
    "Method:\n"
    "1. Call `search_document` with a focused query for the answer.\n"
    "2. If the passages do not contain the answer, call it AGAIN with refined "
    "wording: synonyms, the likely defined term, or a rephrasing (e.g. 'base "
    "currency' → 'agreed currency' → 'currency of the loan'). Keep refining.\n"
    "3. Stop once you have found the answer or are confident it is not stated.\n\n"
    "Then give your FINAL answer as a single JSON object on one line, with keys:\n"
    '  "answer"     : the precise value, or null if not stated in the document.\n'
    '  "references" : a JSON array of EVERY passage that supports the answer — an '
    "answer is often stated in a definition AND again in an operative clause; "
    "include them ALL, not just one. You MUST include the passage that DIRECTLY "
    "STATES the answer value (e.g. the exact text naming the currency / amount / "
    "date), quoted verbatim — do not cite only the surrounding definition or "
    "operative clauses. Each array item is an object:\n"
    '       {"quote": <VERBATIM text copied EXACTLY from a retrieved passage>, '
    '"confidence": "high"|"medium"|"low"}\n'
    '  "confidence" : overall "high" | "medium" | "low".\n'
    '  "reasoning"  : one short sentence.\n\n'
    "Every quote MUST be copied character-for-character from a passage the tool "
    "returned (do not paraphrase, do not invent). If you cannot find any supporting "
    "quote, set answer to null and references to []. Output ONLY the JSON object as "
    "your final message."
)


def _parse_final(text: str) -> dict:
    """Pull the JSON answer object out of the agent's final message."""
    # Prefer a fenced or bare {...} block; fall back to the whole string.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    raw = m.group(0) if m else text
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return {}


def _extract_quotes(data: dict) -> list[tuple[str, str | None]]:
    """
    Normalise the agent's answer into a list of (quote, confidence) pairs,
    accepting both the new `references` array and a legacy singular `reference`.
    """
    pairs: list[tuple[str, str | None]] = []

    refs = data.get("references")
    if isinstance(refs, list):
        for item in refs:
            if isinstance(item, dict):
                quote = item.get("quote") or item.get("reference")
                conf = item.get("confidence")
            else:
                quote, conf = item, None
            if quote:
                pairs.append((str(quote), conf))

    # Back-compat / fallback: a single top-level "reference" string.
    legacy = data.get("reference")
    if legacy and not pairs:
        pairs.append((str(legacy), data.get("confidence")))

    return pairs


_CONF_RANK = {"high": 0, "medium": 1, "low": 2}


def _ground_quotes(
    quote_pairs: list[tuple[str, str | None]],
    pool: dict[int, tuple[Document, float]],
    whole_doc: list[tuple[Document, str]] | None = None,
) -> tuple[list[QueryReference], int]:
    """
    Ground every quote against the retrieved pool (and, as a fallback, the whole
    document), deduping and ranking the survivors (high confidence first, then
    similarity score).

    Returns (references, dropped) where `dropped` counts quotes that did NOT
    ground verbatim — the loop uses that signal to ask the agent to re-quote.
    """
    references: list[QueryReference] = []
    seen: set[str] = set()
    dropped = 0
    for quote, conf in quote_pairs:
        grounded = _ground_reference(quote, pool, whole_doc)
        if grounded is None:
            dropped += 1
            continue
        key = _normalize(quote)
        if key in seen:                      # same proof quoted twice
            continue
        seen.add(key)
        src_doc, src_score = grounded
        references.append(QueryReference(
            reference=str(quote),
            confidence=conf,
            heading=src_doc.metadata.get("heading"),
            pages=src_doc.metadata.get("pages", []),
            chunk_id=src_doc.metadata.get("chunk_id"),
            score=(None if src_score is None else float(src_score)),
        ))
    references.sort(key=lambda r: (_CONF_RANK.get((r.confidence or "").lower(), 3),
                                   -(r.score or 0.0)))
    return references, dropped


# ── Answer-passage augmentation (Option 1) ────────────────────────────────────
#
# The LLM's `references` are the quotes IT chose to cite; it systematically
# under-cites, often quoting the surrounding definition/operative clauses while
# omitting the passage that literally states the value — even when that passage
# was retrieved with the HIGHEST similarity score. This step guarantees the
# answer-stating passage(s) are cited: it scans the retrieved pool (and, as a
# fallback, the whole document) for the answer value and folds the top matches in.

def _answer_variants(answer: str) -> list[str]:
    """
    The answer value plus light variants to look for in passage text:
    the value itself, any parenthetical abbreviation ("Singapore Dollars (SGD)"
    → "SGD"), and the value with the parenthetical stripped. Variants shorter than
    3 normalised chars are dropped to avoid spurious substring hits.
    """
    answer = (answer or "").strip()
    variants = {answer} if answer else set()
    m = re.search(r"\(([^)]+)\)", answer)
    if m:
        variants.add(m.group(1).strip())
    stripped = re.sub(r"\s*\([^)]*\)\s*", " ", answer).strip()
    if stripped:
        variants.add(stripped)
    return [v for v in variants if len(_normalize(v)) >= 3]


def _snippet_containing(text: str, needles: list[str]) -> str:
    """
    Return the first sentence/line of `text` that contains one of `needles`
    (normalised) as a focused verbatim quote; fall back to the whole passage.
    The result is a substring of `text`, so it stays verbatim/grounded.
    """
    for part in re.split(r"\n+|(?<=[.;])\s+", text):
        part_norm = _normalize(part)
        if any(n in part_norm for n in needles) and part.strip():
            return part.strip()
    return text.strip()


def _augment_answer_references(
    answer: str,
    references: list[QueryReference],
    pool: dict[int, tuple[Document, float]],
    whole_doc: list[tuple[Document, str]] | None,
    max_add: int,
) -> int:
    """
    Ensure passages that VERBATIM state the answer value are cited.

    - Marks any already-cited chunk that states the answer with states_answer=True.
    - Appends up to `max_add` highest-scoring un-cited passages that state the
      answer (pool matches, which keep their score, before score-less whole-doc
      matches), each as a focused snippet.

    Returns the number of references added. No-op when the answer value is not
    verbatim anywhere (e.g. a derived value like "Singapore Dollars" when the doc
    only says "the lawful currency of Singapore").
    """
    needles = [_normalize(v) for v in _answer_variants(answer)]
    if not needles:
        return 0

    def _hit(text_norm: str) -> bool:
        return any(n in text_norm for n in needles)

    referenced_chunks = {r.chunk_id for r in references if r.chunk_id is not None}
    ref_by_chunk = {r.chunk_id: r for r in references if r.chunk_id is not None}

    # Answer-stating passages: pool first (scored, keep best score per chunk)…
    matches: dict[int, tuple[Document, float | None]] = {}
    for doc, score in pool.values():
        cid = doc.metadata.get("chunk_id")
        if cid is None or not _hit(_normalize(doc.page_content)):
            continue
        prev = matches.get(cid)
        if prev is None or (prev[1] or -1.0) < (score or -1.0):
            matches[cid] = (doc, score)
    # …then whole-document fallback (score-less) for chunks not already found.
    if whole_doc:
        for doc, norm in whole_doc:
            cid = doc.metadata.get("chunk_id")
            if cid is None or cid in matches:
                continue
            if _hit(norm):
                matches[cid] = (doc, None)

    # Mark cited chunks that state the answer; queue the rest for adding.
    to_add: list[tuple[Document, float | None]] = []
    for cid, (doc, score) in matches.items():
        if cid in referenced_chunks:
            r = ref_by_chunk.get(cid)
            if r is not None:
                r.states_answer = True
        else:
            to_add.append((doc, score))

    # Highest-scoring first; score-less whole-doc matches last.
    to_add.sort(key=lambda ds: (ds[1] is None, -(ds[1] or 0.0)))

    added = 0
    for doc, score in to_add[:max_add]:
        references.append(QueryReference(
            reference=_snippet_containing(doc.page_content, needles),
            confidence="high",
            heading=doc.metadata.get("heading"),
            pages=doc.metadata.get("pages", []),
            chunk_id=doc.metadata.get("chunk_id"),
            score=(None if score is None else float(score)),
            states_answer=True,
        ))
        added += 1
    return added


# ── Tool-calling loop (version-stable: langchain_core + langchain_openai only) ─
#
# langchain 1.x removed AgentExecutor / create_tool_calling_agent from
# langchain.agents. Rather than depend on that churning API, we drive tool calls
# manually with ChatOpenAI.bind_tools — supported across langchain_core versions.

def _coverage_feedback(
    data: dict,
    pool: dict[int, tuple[Document, float]],
    n_distinct_searches: int,
    min_searches: int,
    min_references: int,
    whole_doc: list[tuple[Document, str]] | None = None,
) -> str | None:
    """
    Decide whether the agent's proposed final answer is good enough to accept.

    Returns None to ACCEPT, or a feedback string telling the agent what to fix:
      • too few distinct search angles tried,
      • quotes that did not ground verbatim (re-quote / search again),
      • fewer grounded references than required (look for more — definition AND
        operative clause).
    This is what turns a one-shot loop into a self-correcting one.
    """
    answer = data.get("answer") or None
    quote_pairs = _extract_quotes(data)
    references, dropped = _ground_quotes(quote_pairs, pool, whole_doc)

    problems: list[str] = []

    # Always make the agent try a few genuinely different angles before it is
    # allowed to conclude — including when it wants to say "not found".
    if n_distinct_searches < min_searches:
        problems.append(
            f"You have only run {n_distinct_searches} distinct search(es). Search "
            f"at least {min_searches} DIFFERENT angles (synonyms, the likely "
            f"defined term, related operative clauses) before concluding."
        )

    if answer:
        if dropped:
            problems.append(
                f"{dropped} of your quotes were NOT found verbatim in the "
                f"retrieved passages. Copy each quote character-for-character from "
                f"a passage the tool returned, or search again to locate the exact "
                f"wording."
            )
        if len(references) < min_references:
            problems.append(
                f"You have {len(references)} grounded reference(s) but need at "
                f"least {min_references}. The answer is often stated in BOTH a "
                f"definition and an operative clause — search for the other "
                f"occurrence(s)."
            )

    if not problems:
        return None
    return ("Not finished yet. " + " ".join(problems)
            + " Use search_document again, then give your final JSON.")


def _run_agent_loop(
    question: str,
    search_tool: StructuredTool,
    pool: dict[int, tuple[Document, float]],
    search_terms: list[str],
    max_iterations: int,
    min_searches: int,
    min_references: int,
    max_nudges: int,
    llm_model: str,
    base_url: str,
    api_key: str,
    whole_doc: list[tuple[Document, str]] | None = None,
    key_terms: list[str] | None = None,
) -> dict:
    """
    Self-correcting tool-calling loop.

    The model may call `search_document` repeatedly (refining its queries). When
    it stops and proposes a final answer, the answer is validated IN-LOOP against
    the grounding guard and coverage thresholds (min distinct searches, min
    grounded references). If it falls short, the agent is nudged with specific
    feedback and the loop continues — up to `max_nudges` corrective rounds —
    instead of accepting a thin first answer.

    Returns the parsed final JSON dict (may be {} if the model never produced one).
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools([search_tool])

    human = f"Question: {question}"
    if key_terms:
        human += ("\n\nKey terms detected in this question — search for these "
                  "individually as needed: " + ", ".join(key_terms))
    messages: list = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=human),
    ]

    last_data: dict = {}
    nudges_used = 0
    for _ in range(max_iterations):
        ai: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                try:
                    observation = search_tool.invoke(call["args"])
                except Exception as exc:   # noqa: BLE001 — bad args, etc.
                    observation = f"Tool error: {exc}"
                messages.append(ToolMessage(content=str(observation),
                                            tool_call_id=call["id"]))
            continue

        # No tool call → the model thinks it is done. Validate before accepting.
        text = ai.content if isinstance(ai.content, str) else str(ai.content or "")
        data = _parse_final(text)
        last_data = data or last_data

        n_distinct = len({_normalize(t) for t in search_terms})
        feedback = _coverage_feedback(
            data, pool, n_distinct, min_searches, min_references, whole_doc,
        )
        if feedback is None or nudges_used >= max_nudges:
            break  # accept the answer

        nudges_used += 1
        print(f"      → [loop] nudge {nudges_used}/{max_nudges}: {feedback[:110]}…")
        messages.append(HumanMessage(content=feedback))

    return last_data


# ── Main entry point ──────────────────────────────────────────────────────────

def run_query(
    index: DocumentIndex,
    question: str,
    top_k: int = 6,
    max_iterations: int = 8,
    min_searches: int = 2,
    min_references: int = 1,
    max_nudges: int = 3,
    max_answer_refs: int = 3,
    expand_query: bool = True,
    use_llm_terms: bool = True,
    max_seed_terms: int = 6,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> QueryResult:
    """
    Answer an open natural-language question about the document.

    Strategy (self-correcting agentic retrieve-refine + grounding guard):
    0. Key terms are extracted from the question and each gets a focused retrieval
       seeded into the pool, so a nested term ("Adjusted EBITDA") in a long query
       is not buried by the query's bulk (coverage). The agent is told the terms.
    1. A tool-calling agent drives `search_document`, issuing and REFINING its own
       queries (repeated wording is rejected to force genuine variation).
    2. When it proposes a final answer it is validated IN-LOOP: if it tried too
       few distinct searches, quoted text that does not ground verbatim, or found
       fewer references than required, it is nudged with specific feedback and
       keeps going (up to max_nudges corrective rounds).
    3. Every surviving quote is verified deterministically against the retrieved
       passages and, as a fallback, the whole document; only quotes that are not a
       verbatim (encoding-normalised) match anywhere are dropped.
    4. Answer-passage augmentation: the passage(s) that literally state the answer
       value are guaranteed to be cited (marked states_answer=True and ranked
       first), even if the model under-cited and omitted them — up to
       max_answer_refs high-scoring passages are folded in.

    Args:
        index:          Pre-built DocumentIndex (DocumentIndex.from_pdf(...)).
        question:       Natural-language question, e.g. "What is the base currency?"
        top_k:          Passages returned per search call.
        max_iterations: Hard cap on agent steps (search calls + final answers).
        min_searches:   Distinct search angles required before the agent may conclude.
        min_references: Grounded references required before an answer is accepted.
        max_nudges:     Max corrective feedback rounds when coverage falls short.
        max_answer_refs: Max answer-stating passages to fold in (step 4). 0 disables.
        expand_query:   If True, extract key terms from the question and seed a
                        focused retrieval per term (coverage for long queries).
        use_llm_terms:  If True, augment the heuristic key terms with the LLM
                        extractor; if False, use the deterministic heuristic only.
        max_seed_terms: Max key terms to seed with.

    Returns:
        QueryResult. result.answer is the value; result.references lists EVERY
        grounded supporting quote — passages that literally state the answer
        (states_answer=True) ranked first, then by confidence and score — with the
        best mirrored into result.reference/heading/pages for single-result callers.
        result.supports lists the retrieved passages. found=False (references
        empty) is the "No" case — nothing could be grounded.
    """
    print(f"\n[query] Answering: '{question}'")

    pool: dict[int, tuple[Document, float]] = {}
    search_terms: list[str] = []
    search_tool = _make_search_tool(index, pool, search_terms, top_k)

    # Whole-document haystack for the grounding fallback (F): normalise every
    # chunk once so a genuine quote from a passage the agent never retrieved can
    # still be grounded. Precomputed here to avoid re-normalising on each check.
    whole_doc: list[tuple[Document, str]] = [
        (doc, _normalize(doc.page_content))
        for doc in (chunk_to_doc(c) for c in index.chunks)
    ]

    # Extract key terms so a nested term ("Adjusted EBITDA") gets its own focused
    # search — otherwise a long query's embedding may bury it below the top_k.
    key_terms: list[str] = []
    if expand_query:
        key_terms = extract_query_terms(
            question, use_llm=use_llm_terms, max_terms=max_seed_terms,
            llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        if key_terms:
            print(f"      → Key terms for seeding: {key_terms}")

    # Seed the pool with the full question AND each key term, so we have strong
    # candidates to ground against / fall back on even if the agent never calls the
    # tool. A focused term retrieval scores its chunk higher than the long-query
    # retrieval does; keep the MAX score per chunk so that higher score wins.
    for seed_query in [question, *key_terms]:
        for doc, score in retrieve_content_candidates(index, seed_query, top_k=top_k):
            usable = materialize_candidate(doc, index)
            if usable is None:
                continue
            cid = usable.metadata.get("chunk_id")
            if cid is not None and (cid not in pool or score > pool[cid][1]):
                pool[cid] = (usable, score)

    data: dict = {}
    try:
        data = _run_agent_loop(
            question, search_tool, pool, search_terms,
            max_iterations=max_iterations, min_searches=min_searches,
            min_references=min_references, max_nudges=max_nudges,
            llm_model=llm_model, base_url=base_url, api_key=api_key,
            whole_doc=whole_doc, key_terms=key_terms,
        )
        print(f"      → Agent searched {len(search_terms)} time(s): {search_terms}")
        if not data:   # backend ignored tools / produced no JSON → single-shot
            print("      → No structured answer from loop; trying single-shot")
            data = _single_shot_fallback(
                question, pool, llm_model=llm_model, base_url=base_url, api_key=api_key,
            )

    except Exception as exc:   # noqa: BLE001 — tool-calling unsupported, etc.
        print(f"      → Agent loop failed ({type(exc).__name__}: {exc}); "
              f"answering from seed retrieval only")
        data = _single_shot_fallback(
            question, pool, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )

    answer = (data.get("answer") or None)
    confidence = data.get("confidence")
    reasoning = data.get("reasoning")
    quote_pairs = _extract_quotes(data)

    supports = sorted(
        (_to_match(d, s, None) for d, s in pool.values()),
        key=lambda m: -(m.score or 0.0),
    )

    base = dict(question=question, key_terms=key_terms, search_terms=search_terms,
                supports=supports, reasoning=reasoning, confidence=confidence)

    # ── Grounding guard: every quote must be verbatim in a document passage ──
    if not answer:
        print("      → No answer produced → Not found")
        return QueryResult(found=False, **base)

    references, dropped = _ground_quotes(quote_pairs, pool, whole_doc)
    if dropped:
        print(f"      → Dropped {dropped} ungrounded quote(s) (possible hallucination)")

    # Option 1: guarantee the passage(s) that literally state the answer are cited,
    # even if the model under-cited and omitted the highest-scoring one.
    if max_answer_refs > 0:
        added = _augment_answer_references(
            answer, references, pool, whole_doc, max_answer_refs,
        )
        if added:
            print(f"      → Added {added} answer-stating passage(s) the model omitted")

    # Rank: answer-stating passages first, then confidence, then similarity score.
    references.sort(key=lambda r: (not r.states_answer,
                                   _CONF_RANK.get((r.confidence or "").lower(), 3),
                                   -(r.score or 0.0)))

    if not references:
        print("      → No quote grounded anywhere in the document → Not found")
        return QueryResult(found=False, **{**base, "confidence": "low"})

    best = references[0]
    print(f"      → Answer: {answer!r}  ({len(references)} grounded reference(s); "
          f"best in {best.heading!r}, confidence={best.confidence})")
    return QueryResult(
        found=True,
        answer=str(answer),
        references=references,
        # Mirror the best reference into the flat convenience fields.
        reference=best.reference,
        heading=best.heading,
        pages=best.pages,
        chunk_id=best.chunk_id,
        score=best.score,
        **{k: v for k, v in base.items() if k != "confidence"},
        confidence=confidence,
    )


# ── Fallback for backends without tool calling ────────────────────────────────

def _single_shot_fallback(
    question: str,
    pool: dict[int, tuple[Document, float]],
    llm_model: str,
    base_url: str,
    api_key: str,
) -> dict:
    """
    No agent loop: show the seed passages once and ask for the same JSON answer.
    The grounding guard in run_query still applies to whatever this returns.
    """
    if not pool:
        return {}
    passages = sorted(pool.values(), key=lambda x: -x[1])
    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('heading')}]\n{d.page_content.strip()[:1200]}"
        for d, _ in passages
    )
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)
    # Plain messages, NOT ChatPromptTemplate: _SYSTEM contains literal JSON braces
    # ('{"quote": ...}') which a template would misread as input variables.
    system_text = _SYSTEM.replace(
        "You cannot see the document directly — you must use the "
        "`search_document` tool to retrieve passages.",
        "You are given the most relevant passages below.")
    out = llm.invoke([
        SystemMessage(content=system_text),
        HumanMessage(content=f"Question: {question}\n\nPassages:\n{context}"),
    ]).content
    return _parse_final(str(out))


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.query <pdf_path> <question>")
        raise SystemExit(1)

    idx = DocumentIndex.from_pdf(sys.argv[1])
    res = run_query(idx, sys.argv[2])
    print("\n" + "=" * 60)
    print("ANSWER    :", res.answer if res.found else "(not found)")
    print("KEY TERMS :", res.key_terms)
    print("CONFIDENCE:", res.confidence)
    print(f"REFERENCES: {len(res.references)} grounded")
    for i, ref in enumerate(res.references, 1):
        star = " ★states-answer" if ref.states_answer else ""
        score = "n/a" if ref.score is None else f"{ref.score:.3f}"
        print(f"  [{i}] ({ref.confidence}, score={score}){star} {ref.heading} {ref.pages}")
        print(f"      {ref.reference[:200]}")
    print(f"SUPPORTS  : {len(res.supports)} retrieved passage(s)")
    for i, sup in enumerate(res.supports, 1):
        score = "n/a" if sup.score is None else f"{sup.score:.3f}"
        print(f"  [{i}] (score={score}) {sup.heading} {sup.pages}")
    print("TRIED     :", res.search_terms)
