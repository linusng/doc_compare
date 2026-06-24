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
from .models import ContentMatch, ContentMatchResult
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

def _to_match(doc: Document, score: float, llm_output: str | None) -> ContentMatch:
    return ContentMatch(
        content=doc.page_content,
        heading=doc.metadata.get("heading"),
        pages=doc.metadata.get("pages", []),
        chunk_id=doc.metadata.get("chunk_id"),
        score=float(score),
        llm_output=llm_output,
        expanded=bool(doc.metadata.get("expanded_from_heading")),
    )


def _build_result(query: str, matches: list[ContentMatch],
                  best_score: float | None) -> ContentMatchResult:
    """Wrap the confirmed matches, mirroring the best one in the top-level fields."""
    if not matches:
        return ContentMatchResult(query=query, found=False, score=best_score)
    best = matches[0]
    return ContentMatchResult(
        query=query,
        found=True,
        content=best.content,
        heading=best.heading,
        pages=best.pages,
        chunk_id=best.chunk_id,
        score=best.score,
        llm_output=best.llm_output,
        matches=matches,
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
    Find the passage(s) of content that match a free-text query.

    Strategy:
    1. Rank chunks by semantic similarity to the query.
    2. If verify=True, evaluate EVERY candidate (expanding header-only section
       containers to their body, skipping candidates with no write-up) and keep
       all that the LLM confirms — not just the first. Confirmed matches are
       ranked best (highest similarity) first.
    3. If nothing is retrieved/confirmed, return found=False (the "No" case).

    Args:
        index:     Pre-built DocumentIndex (DocumentIndex.from_pdf(...)).
        query:     Arbitrary free-text to match against the document content.
        verify:    If True, LLM-confirm every candidate and return all matches.
                   If False, return the single highest-similarity passage.
        top_k:     How many candidate chunks to retrieve/consider.
        min_score: Optional similarity floor. Candidates scoring below it are not
                   considered; if none clear it, the result is No.

    Returns:
        ContentMatchResult. `result.matches` lists every confirmed passage,
        best-first; the top-level fields (content/heading/…) mirror matches[0].
        result.found / result.answer indicate Yes/No.
    """
    print(f"\n[find] Searching content for: '{query}'")

    candidates = retrieve_content_candidates(index, query, top_k=top_k)
    if not candidates:
        print("      → No candidates retrieved → No")
        return ContentMatchResult(query=query, found=False)

    best_doc, best_score = candidates[0]
    print(f"      → {len(candidates)} candidates (best score {best_score:.4f}: "
          f"{best_doc.metadata.get('heading')!r})")

    # Optional hard similarity floor — drop candidates not close enough to bother.
    if min_score is not None:
        candidates = [(d, s) for d, s in candidates if s >= min_score]
        if not candidates:
            print(f"      → No candidate ≥ min_score {min_score} → No")
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
            return _build_result(query, [_to_match(usable, score, None)], float(best_score))
        print("      → No candidate had a write-up → No")
        return ContentMatchResult(query=query, found=False, score=float(best_score))

    # Evaluate EVERY candidate and collect all confirmed matches. Header-only
    # containers are expanded to their section body before confirmation;
    # candidates with no write-up are skipped.
    print("[find] Verifying all candidates...")
    confirmed_matches: list[ContentMatch] = []
    seen_content: set[str] = set()

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
            key = usable.page_content.strip()
            if key in seen_content:        # skip exact-duplicate passages
                continue
            seen_content.add(key)
            confirmed_matches.append(_to_match(usable, score, llm_output))

    if not confirmed_matches:
        print("      → No candidate confirmed → No")
        return ContentMatchResult(query=query, found=False, score=float(best_score))

    # Rank confirmed matches best (highest similarity) first.
    confirmed_matches.sort(key=lambda m: -(m.score or 0.0))
    print(f"      → {len(confirmed_matches)} confirmed match(es); "
          f"best: {confirmed_matches[0].heading!r}")
    return _build_result(query, confirmed_matches, float(best_score))
