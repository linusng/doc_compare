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
from .section_utils import (
    _extract_named_prefix,
    _extract_section_number,
    gather_full_section,
    merge_section_chunks,
)


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


# ── Candidate materialisation (avoid header-only matches) ─────────────────────

def _body_len(doc: Document) -> int:
    """Length of the chunk's body (content), excluding the heading line."""
    n = doc.metadata.get("content_length")
    if n is not None:
        return int(n)
    # Fallback: page_content is "<heading>\n<content>"; subtract the heading.
    heading = doc.metadata.get("heading", "") or ""
    return max(0, len(doc.page_content.strip()) - len(heading.strip()))


def _is_header_only_container(doc: Document) -> bool:
    """
    True if the candidate is a section *title* with no body of its own —
    e.g. '5. INSURANCE' or 'Schedule 2 …' kept by filter_short_chunks as a
    navigation anchor. Its page_content is just the heading, so returning it
    gives no write-up.

    A defined-term chunk whose meaning lives in the heading itself (e.g.
    '"Margin" means 0.80 per cent.') is NOT flagged: it has no leading section
    number / named prefix, so the heading IS the content.
    """
    if _body_len(doc) > 0:
        return False
    heading = doc.metadata.get("heading", "") or ""
    return (
        _extract_section_number(heading) is not None
        or _extract_named_prefix(heading) is not None
    )


def materialize_candidate(
    doc: Document,
    index: DocumentIndex,
) -> Document | None:
    """
    Return a Document whose page_content holds a real write-up.

    - Normal body chunk (or content-bearing heading): returned as-is.
    - Header-only section container: expanded to the full section body via
      gather_full_section + merge_section_chunks, so the write-up is included.
    - No write-up anywhere (empty container with no gatherable body): None, so
      the caller skips it and looks at the next candidate.
    """
    if not _is_header_only_container(doc):
        return doc if doc.page_content.strip() else None

    section_chunks = gather_full_section(doc, index.chunks)
    body_chars = sum(len(c.content.strip()) for c in section_chunks)
    if body_chars == 0:
        return None  # truly nothing beneath this title

    merged = merge_section_chunks(section_chunks)
    return Document(
        page_content=merged.content,
        metadata={
            **doc.metadata,
            "heading": merged.heading,
            "pages": merged.pages,
            "content_length": len(merged.content),
            "expanded_from_heading": True,
        },
    )


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

    # Without verification, return the closest passage that has a real write-up
    # (expanding a header-only section container into its body, skipping any
    # candidate with no write-up at all).
    if not verify:
        for doc, score in candidates:
            usable = materialize_candidate(doc, index)
            if usable is None:
                print(f"      → Skipping header-only candidate "
                      f"{doc.metadata.get('heading')!r} (no write-up)")
                continue
            tag = " (expanded to section body)" if usable.metadata.get("expanded_from_heading") else ""
            print(f"      → Returning closest passage{tag}: "
                  f"{usable.metadata.get('heading')!r}")
            return _to_result(query, usable, score, llm_output=None, found=True)
        print("      → No candidate had a write-up → No")
        return ContentMatchResult(query=query, found=False, score=float(best_score))

    # Verify loop: first confirmed candidate WITH a real write-up wins. Header-
    # only containers are expanded to their section body before confirmation;
    # candidates with no write-up are skipped so the loop reaches real content
    # instead of stopping on a bare heading at attempt 1.
    print("[find] Verifying candidates...")
    for attempt, (doc, score) in enumerate(candidates, start=1):
        heading = doc.metadata.get("heading", "?")

        usable = materialize_candidate(doc, index)
        if usable is None:
            print(f"      → [diag] attempt {attempt}: score={score:.4f}, "
                  f"heading={heading!r} → skipped (header-only, no write-up)")
            continue

        expanded = bool(usable.metadata.get("expanded_from_heading"))
        confirmed, llm_output = confirm_content_match(
            usable, query, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        print(f"      → [diag] attempt {attempt}: score={score:.4f}, "
              f"heading={usable.metadata.get('heading')!r}, "
              f"body_chars={len(usable.page_content)}, expanded={expanded}, "
              f"confirmed={confirmed}")

        if confirmed:
            print(f"      → Matched on attempt {attempt}: "
                  f"{usable.metadata.get('heading')!r}"
                  f"{' (expanded to section body)' if expanded else ''}")
            return _to_result(query, usable, score, llm_output=llm_output, found=True)

    # Nothing confirmed across all candidates → No.
    print("      → No candidate confirmed → No")
    return ContentMatchResult(query=query, found=False, score=float(best_score))
