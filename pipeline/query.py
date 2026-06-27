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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import ContentMatch, QueryResult
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
    '  "reference"  : a VERBATIM quote copied EXACTLY from a retrieved passage '
    "that contains the answer (do not paraphrase, do not invent), or null.\n"
    '  "confidence" : "high" | "medium" | "low".\n'
    '  "reasoning"  : one short sentence.\n\n'
    "The reference MUST be copied character-for-character from a passage the tool "
    "returned. If you cannot find a supporting quote, set answer and reference to "
    "null. Output ONLY the JSON object as your final message."
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


# ── Main entry point ──────────────────────────────────────────────────────────

def run_query(
    index: DocumentIndex,
    question: str,
    top_k: int = 6,
    max_iterations: int = 5,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> QueryResult:
    """
    Answer an open natural-language question about the document.

    Strategy (agentic retrieve-refine + grounding guard):
    1. A tool-calling agent drives `search_document`, issuing and REFINING its own
       queries until it can answer or gives up (capped at max_iterations).
    2. The agent returns answer + a verbatim reference quote + confidence.
    3. The reference is verified deterministically against the passages actually
       retrieved; an ungrounded (hallucinated/paraphrased) quote → found=False.

    Args:
        index:          Pre-built DocumentIndex (DocumentIndex.from_pdf(...)).
        question:       Natural-language question, e.g. "What is the base currency?"
        top_k:          Passages returned per search call.
        max_iterations: Max agent steps (search calls + final answer).

    Returns:
        QueryResult. result.found / result.answer / result.reference carry the
        grounded answer; result.supports lists the retrieved passages; result
        is the "No" case (found=False, answer=None) when nothing is grounded.
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
        from langchain.agents import AgentExecutor, create_tool_calling_agent

        llm = ChatOpenAI(
            model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM),
            ("human", "Question: {question}"),
            ("placeholder", "{agent_scratchpad}"),
        ])
        agent = create_tool_calling_agent(llm, [search_tool], prompt)
        executor = AgentExecutor(
            agent=agent, tools=[search_tool], verbose=False,
            max_iterations=max_iterations, handle_parsing_errors=True,
        )
        result = executor.invoke({"question": question})
        data = _parse_final(str(result.get("output", "")))
        print(f"      → Agent searched {len(search_terms)} time(s): {search_terms}")

    except Exception as exc:   # noqa: BLE001 — tool-calling unsupported, etc.
        print(f"      → Agent unavailable ({type(exc).__name__}: {exc}); "
              f"answering from seed retrieval only")
        data = _single_shot_fallback(
            question, pool, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )

    answer = (data.get("answer") or None)
    reference = (data.get("reference") or None)
    confidence = data.get("confidence")
    reasoning = data.get("reasoning")

    supports = sorted(
        (_to_match(d, s, None) for d, s in pool.values()),
        key=lambda m: -(m.score or 0.0),
    )

    # ── Grounding guard: the reference must be verbatim in a retrieved passage ──
    if not answer or not reference:
        print("      → No answer/reference produced → Not found")
        return QueryResult(question=question, found=False, search_terms=search_terms,
                           supports=supports, reasoning=reasoning, confidence=confidence)

    grounded = _ground_reference(reference, pool)
    if grounded is None:
        print(f"      → Reference not grounded in any retrieved passage "
              f"(possible hallucination): {reference[:80]!r} → Not found")
        return QueryResult(question=question, found=False, search_terms=search_terms,
                           supports=supports, reasoning=reasoning, confidence="low")

    src_doc, src_score = grounded
    print(f"      → Answer: {answer!r}  (grounded in "
          f"{src_doc.metadata.get('heading')!r}, confidence={confidence})")
    return QueryResult(
        question=question,
        found=True,
        answer=str(answer),
        reference=str(reference),
        heading=src_doc.metadata.get("heading"),
        pages=src_doc.metadata.get("pages", []),
        chunk_id=src_doc.metadata.get("chunk_id"),
        score=float(src_score),
        confidence=confidence,
        reasoning=reasoning,
        search_terms=search_terms,
        supports=supports,
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
    print("ANSWER   :", res.answer if res.found else "(not found)")
    print("REFERENCE:", res.reference)
    print("HEADING  :", res.heading, res.pages)
    print("CONFIDENCE:", res.confidence)
    print("TRIED    :", res.search_terms)
