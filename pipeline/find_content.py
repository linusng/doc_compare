"""
Find mode: locate the passage of content that best matches a FREE-TEXT query.

Main entry point: run_find(index, query, ...)

This mirrors pipeline.extract.run_extract, but instead of resolving a named
section it does a content search: it ranks chunks by semantic similarity to an
arbitrary free-text string and, when verify=True, walks the ranked list asking
the LLM to confirm each candidate actually matches. The first confirmed passage
is returned. If nothing is confirmed (or nothing is retrieved), the result has
found=False — the "return No" case.

Key difference from run_extract: it returns the closest *part* of the content
(the best-matching chunk), NOT the whole reassembled section. There is therefore
no gather_full_section / merge step.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import ContentMatchResult


# ── Semantic retrieval ────────────────────────────────────────────────────────

def retrieve_content_candidates(
    index: DocumentIndex,
    query: str,
    top_k: int = 8,
) -> list[tuple[Document, float]]:
    """
    Rank chunks by semantic similarity to a free-text query.

    Unlike extract.retrieve_candidates this adds no "Section {q}" angles and no
    heading-substring injection — the query is treated as arbitrary content, not
    a section name. Returns (doc, score) deduped by chunk_id, best score first
    (higher score = more similar, matching the rest of the pipeline).
    """
    seen: dict[int, tuple[Document, float]] = {}
    for doc, score in index.vector_store.similarity_search_with_score(query, k=top_k):
        cid = doc.metadata["chunk_id"]
        if cid not in seen or score > seen[cid][1]:
            seen[cid] = (doc, score)
    return sorted(seen.values(), key=lambda x: -x[1])


# ── LLM confirmation ──────────────────────────────────────────────────────────

def confirm_content_match(
    doc: Document,
    query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> tuple[bool, str]:
    """
    Ask the LLM whether the candidate passage actually matches the free-text
    query — i.e. the passage contains or addresses what the query describes, not
    merely sharing a few surface words.

    Pure yes/no judgement: it never rewrites the passage (the returned content is
    always the verbatim chunk). Returns (is_confirmed, raw_llm_answer); the raw
    answer is kept for audit only.
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst. You are given a free-text "
            "description and a candidate passage from a legal agreement.\n\n"
            "Decide whether the passage actually contains or addresses what the "
            "description refers to — not merely sharing a few words.\n\n"
            "Respond with EXACTLY one word and nothing else:\n"
            "- YES if the passage matches the description.\n"
            "- NO if it does not."
        )),
        ("human", "Description:\n{query}\n\nPassage:\n{text}"),
    ])

    output = (prompt | llm).invoke(
        {"query": query, "text": doc.page_content[:4000]}
    ).content.strip()
    confirmed = output.strip().upper().startswith("YES")
    return confirmed, output


# ── Main entry point ──────────────────────────────────────────────────────────

def _to_result(query: str, doc: Document, score: float, llm_output: str | None,
               found: bool) -> ContentMatchResult:
    return ContentMatchResult(
        query=query,
        found=found,
        content=doc.page_content if found else None,
        heading=doc.metadata.get("heading"),
        pages=doc.metadata.get("pages", []),
        chunk_id=doc.metadata.get("chunk_id"),
        score=float(score),
        llm_output=llm_output,
    )


def run_find(
    index: DocumentIndex,
    query: str,
    verify: bool = True,
    top_k: int = 8,
    min_score: float | None = None,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> ContentMatchResult:
    """
    Find the passage of content that best matches a free-text query.

    Strategy (mirrors run_extract):
    1. Rank chunks by semantic similarity to the query.
    2. If verify=True, walk the ranked list asking the LLM to confirm each
       candidate; return the first confirmed passage.
    3. If nothing is retrieved/confirmed, return found=False (the "No" case).

    Args:
        index:     Pre-built DocumentIndex (DocumentIndex.from_pdf(...)).
        query:     Arbitrary free-text to match against the document content.
        verify:    If True, LLM-confirm candidates in ranked order (recommended).
                   If False, return the single highest-similarity passage.
        top_k:     How many candidate chunks to retrieve/consider.
        min_score: Optional similarity floor. If the best candidate scores below
                   this, return found=False without calling the LLM.

    Returns:
        ContentMatchResult. result.found / result.answer indicate Yes/No;
        result.content holds the matched passage when found.
    """
    print(f"\n[find] Searching content for: '{query}'")

    candidates = retrieve_content_candidates(index, query, top_k=top_k)
    if not candidates:
        print("      → No candidates retrieved → No")
        return ContentMatchResult(query=query, found=False)

    best_doc, best_score = candidates[0]
    print(f"      → {len(candidates)} candidates (best score {best_score:.4f}: "
          f"{best_doc.metadata.get('heading')!r})")

    # Optional hard similarity floor — nothing close enough to bother verifying.
    if min_score is not None and best_score < min_score:
        print(f"      → Best score {best_score:.4f} < min_score {min_score} → No")
        return ContentMatchResult(query=query, found=False, score=float(best_score))

    # Without verification, return the single closest passage.
    if not verify:
        print(f"      → Returning closest passage: {best_doc.metadata.get('heading')!r}")
        return _to_result(query, best_doc, best_score, llm_output=None, found=True)

    # Verify loop: first confirmed candidate wins.
    print("[find] Verifying candidates...")
    for attempt, (doc, score) in enumerate(candidates, start=1):
        heading = doc.metadata.get("heading", "?")
        confirmed, llm_output = confirm_content_match(
            doc, query, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        print(f"      → [diag] attempt {attempt}: score={score:.4f}, "
              f"heading={heading!r}, confirmed={confirmed}")

        if confirmed:
            print(f"      → Matched on attempt {attempt}: {heading!r}")
            return _to_result(query, doc, score, llm_output=llm_output, found=True)

    # Nothing confirmed across all candidates → No.
    print("      → No candidate confirmed → No")
    return ContentMatchResult(query=query, found=False, score=float(best_score))
