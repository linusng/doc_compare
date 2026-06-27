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
Whatever the agent answers, the returned `reference` is checked DETERMINISTICALLY
against the passages that were actually retrieved: it must be a (whitespace-
normalised) verbatim substring of one of them. If it is not, the answer is not
grounded and the result is found=False. This mirrors evidence.py's "the agent
decides, the code enforces truth" guarantee and extract.py's rule that the LLM
never invents document text.
"""

import json
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import ContentMatch, QueryReference, QueryResult
from .find_content import (
    materialize_candidate,
    retrieve_content_candidates,
    _to_match,
)


# ── Grounding (deterministic verbatim check) ──────────────────────────────────

def _normalize(text: str) -> str:
    """Collapse whitespace and lowercase so a quote survives reflow/casing."""
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _ground_reference(
    reference: str,
    pool: dict[int, tuple[Document, float]],
) -> tuple[Document, float] | None:
    """
    Return the (doc, score) whose text verbatim-contains `reference`
    (whitespace-insensitive), or None if the quote is not grounded anywhere.

    The longest reference that still matches wins implicitly: we only accept an
    exact normalised substring, so a hallucinated/paraphrased quote fails.
    """
    needle = _normalize(reference)
    if len(needle) < 4:        # too short to be a meaningful, unique proof
        return None
    best: tuple[Document, float] | None = None
    for doc, score in pool.values():
        if needle in _normalize(doc.page_content):
            if best is None or score > best[1]:
                best = (doc, score)
    return best


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
    "include them ALL, not just one. Each array item is an object:\n"
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
) -> tuple[list[QueryReference], int]:
    """
    Ground every quote against the retrieved pool, deduping and ranking the
    survivors (high confidence first, then similarity score).

    Returns (references, dropped) where `dropped` counts quotes that did NOT
    ground verbatim — the loop uses that signal to ask the agent to re-quote.
    """
    references: list[QueryReference] = []
    seen: set[str] = set()
    dropped = 0
    for quote, conf in quote_pairs:
        grounded = _ground_reference(quote, pool)
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
            score=float(src_score),
        ))
    references.sort(key=lambda r: (_CONF_RANK.get((r.confidence or "").lower(), 3),
                                   -(r.score or 0.0)))
    return references, dropped


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
    references, dropped = _ground_quotes(quote_pairs, pool)

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

    messages: list = [
        SystemMessage(content=_SYSTEM),
        HumanMessage(content=f"Question: {question}"),
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
            data, pool, n_distinct, min_searches, min_references,
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
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> QueryResult:
    """
    Answer an open natural-language question about the document.

    Strategy (self-correcting agentic retrieve-refine + grounding guard):
    1. A tool-calling agent drives `search_document`, issuing and REFINING its own
       queries (repeated wording is rejected to force genuine variation).
    2. When it proposes a final answer it is validated IN-LOOP: if it tried too
       few distinct searches, quoted text that does not ground verbatim, or found
       fewer references than required, it is nudged with specific feedback and
       keeps going (up to max_nudges corrective rounds).
    3. Every surviving quote is verified deterministically against the retrieved
       passages; ungrounded (hallucinated/paraphrased) quotes are dropped.

    Args:
        index:          Pre-built DocumentIndex (DocumentIndex.from_pdf(...)).
        question:       Natural-language question, e.g. "What is the base currency?"
        top_k:          Passages returned per search call.
        max_iterations: Hard cap on agent steps (search calls + final answers).
        min_searches:   Distinct search angles required before the agent may conclude.
        min_references: Grounded references required before an answer is accepted.
        max_nudges:     Max corrective feedback rounds when coverage falls short.

    Returns:
        QueryResult. result.answer is the value; result.references lists EVERY
        grounded supporting quote (ranked high-confidence first), with the best
        mirrored into result.reference/heading/pages for single-result callers.
        result.supports lists the retrieved passages. found=False (references
        empty) is the "No" case — nothing could be grounded.
    """
    print(f"\n[query] Answering: '{question}'")

    pool: dict[int, tuple[Document, float]] = {}
    search_terms: list[str] = []
    search_tool = _make_search_tool(index, pool, search_terms, top_k)

    # Seed the pool so we still have candidates to ground against / fall back on
    # even if the agent never calls the tool.
    for doc, score in retrieve_content_candidates(index, question, top_k=top_k):
        usable = materialize_candidate(doc, index)
        if usable is None:
            continue
        cid = usable.metadata.get("chunk_id")
        if cid is not None and cid not in pool:
            pool[cid] = (usable, score)

    data: dict = {}
    try:
        data = _run_agent_loop(
            question, search_tool, pool, search_terms,
            max_iterations=max_iterations, min_searches=min_searches,
            min_references=min_references, max_nudges=max_nudges,
            llm_model=llm_model, base_url=base_url, api_key=api_key,
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

    base = dict(question=question, search_terms=search_terms, supports=supports,
                reasoning=reasoning, confidence=confidence)

    # ── Grounding guard: every quote must be verbatim in a retrieved passage ──
    if not answer or not quote_pairs:
        print("      → No answer/references produced → Not found")
        return QueryResult(found=False, **base)

    references, dropped = _ground_quotes(quote_pairs, pool)
    if dropped:
        print(f"      → Dropped {dropped} ungrounded quote(s) (possible hallucination)")

    if not references:
        print("      → No quote grounded in any retrieved passage → Not found")
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
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM.replace(
            "You cannot see the document directly — you must use the "
            "`search_document` tool to retrieve passages.",
            "You are given the most relevant passages below.")),
        ("human", "Question: {question}\n\nPassages:\n{context}"),
    ])
    out = (prompt | llm).invoke({"question": question, "context": context}).content
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
    print("CONFIDENCE:", res.confidence)
    print(f"REFERENCES: {len(res.references)} grounded")
    for i, ref in enumerate(res.references, 1):
        print(f"  [{i}] ({ref.confidence}) {ref.heading} {ref.pages}")
        print(f"      {ref.reference[:200]}")
    print("TRIED     :", res.search_terms)
