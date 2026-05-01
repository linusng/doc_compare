"""
Extract mode: find and return one named section from a DocumentIndex.

Main entry point: run_extract(index, section_query, ...)

The function builds a ranked candidate list (heading-matched first, semantic
fallbacks after) and, when verify=True, iterates through it until the LLM
confirms a match or all candidates are exhausted.
"""

import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import ExtractionResult, pydantic_copy
from .section_utils import (
    _base_heading,
    chunk_to_doc,
    gather_full_section,
    merge_section_chunks,
)


# ── Heading-first candidate selection ─────────────────────────────────────────

def select_heading_with_llm(
    index: DocumentIndex,
    section_query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    top_n: int = 3,
) -> list[Document]:
    """
    Present every unique section heading to the LLM and ask it to rank
    up to top_n headings that best match section_query.

    Shows only headings (no body text) so the LLM selects by section title,
    not by which chunk happens to mention the query terms in its body.

    Returns Documents in ranked order; may be empty if no heading matches.
    """
    seen_bases: dict[str, object] = {}
    for chunk in index.chunks:
        base = _base_heading(chunk.heading)
        if base and base not in seen_bases:
            seen_bases[base] = chunk

    unique_chunks = list(seen_bases.values())
    if not unique_chunks:
        return []

    heading_lines = [
        f"[{i}] {_base_heading(c.heading)}"
        for i, c in enumerate(unique_chunks)
    ]
    headings_text = "\n".join(heading_lines)

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are given a numbered list of section headings from a legal "
            "document. Identify which headings best match the following "
            f"section description:\n\n  \"{section_query}\"\n\n"
            "Rules:\n"
            "- Match by heading text only — ignore body content.\n"
            "- The section number in the query is a hint; if the title "
            "matches but the number differs slightly, still select it.\n"
            f"- Return up to {top_n} matches ranked best first.\n\n"
            "Respond with ONLY a comma-separated list of numbers, best first, "
            "e.g.: 5,2,8\n"
            "If no heading matches, respond with: NONE"
        )),
        ("human", "{headings}"),
    ])

    answer = (prompt | llm).invoke({"headings": headings_text}).content.strip()

    if answer.upper() == "NONE":
        return []

    docs: list[Document] = []
    seen_ids: set[int] = set()
    for m in re.finditer(r'\d+', answer):
        idx = int(m.group())
        if 0 <= idx < len(unique_chunks) and idx not in seen_ids:
            docs.append(chunk_to_doc(unique_chunks[idx]))
            seen_ids.add(idx)
        if len(docs) >= top_n:
            break

    return docs


# ── Semantic fallback retrieval ───────────────────────────────────────────────

def retrieve_candidates(
    index: DocumentIndex,
    section_query: str,
    top_k: int = 8,
) -> list[tuple[Document, float]]:
    """
    Hybrid semantic search used as fallback when heading selection fails or
    to supplement it with additional candidates.

    Runs three query angles through the vector store and injects any chunk
    whose heading contains the query text at score=1.0, guaranteeing it
    appears regardless of embedding similarity.
    """
    seen: dict[int, tuple[Document, float]] = {}

    for query in [
        section_query,
        f"Section {section_query}",
        f"{section_query} clause provisions",
    ]:
        for doc, score in index.vector_store.similarity_search_with_score(query, k=top_k):
            cid = doc.metadata["chunk_id"]
            if cid not in seen or score > seen[cid][1]:
                seen[cid] = (doc, score)

    query_lower = section_query.lower()
    for chunk in index.chunks:
        if query_lower in chunk.heading.lower() and chunk.chunk_id not in seen:
            seen[chunk.chunk_id] = (chunk_to_doc(chunk), 1.0)

    return sorted(seen.values(), key=lambda x: -x[1])


# ── LLM re-ranking ────────────────────────────────────────────────────────────

def pick_best_with_llm(
    candidates: list[tuple[Document, float]],
    section_query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> Document:
    """
    Present the top semantic candidates to the LLM and ask it to pick the
    one that best matches section_query. Falls back to the highest-scoring
    embedding candidate if the LLM returns an invalid choice.
    """
    top = candidates[:8]

    options = []
    for i, (doc, _) in enumerate(top):
        heading = doc.metadata.get("heading", "(no heading)")
        preview = doc.page_content[:300].replace("\n", " ")
        options.append(f"[{i}] Heading: {heading}\n    Preview: {preview}...")

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst. You will be shown candidate "
            "sections from a legal agreement. Identify which candidate best "
            f"matches:\n\n  \"{section_query}\"\n\n"
            "Consider both heading and content preview. Section numbers are "
            "hints, not exact requirements — match by meaning.\n\n"
            "Respond with ONLY the number in square brackets, e.g.: 0\n"
            "If none matches, respond with: NONE"
        )),
        ("human", "{options}"),
    ])

    answer = (prompt | llm).invoke({"options": "\n\n".join(options)}).content.strip()

    m = re.search(r'\d+', answer)
    if m:
        idx = int(m.group())
        if 0 <= idx < len(top):
            return top[idx][0]

    return top[0][0]


# ── LLM verification ──────────────────────────────────────────────────────────

def verify_with_llm(
    doc: Document,
    section_query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> tuple[bool, str]:
    """
    Ask the LLM to confirm the retrieved chunk matches section_query.
    Returns (is_confirmed, llm_output).
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document parser. "
            f"Given a candidate text block, confirm whether it is the "
            f'"{section_query}" section. '
            "If yes, return only the cleaned section text. "
            "If no, respond with exactly: NOT_FOUND"
        )),
        ("human", "{text}"),
    ])

    output = (prompt | llm).invoke({"text": doc.page_content[:4000]}).content.strip()
    return output != "NOT_FOUND", output


# ── Main entry point ──────────────────────────────────────────────────────────

def run_extract(
    index: DocumentIndex,
    section_query: str,
    verify: bool = False,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> ExtractionResult:
    """
    Find and return one named section from a pre-built DocumentIndex.

    Strategy:
    1. Build a ranked candidate list:
       - Primary: LLM selects from heading list (section-strict, no body noise).
       - Fallback: hybrid semantic + keyword search appended after.
    2. Retry loop: gather → merge → verify each candidate until confirmed or
       all candidates are exhausted.

    Args:
        index:         Pre-built DocumentIndex (call DocumentIndex.from_pdf first).
        section_query: Section to find, e.g. "3.1 Purpose" or "Schedule 2".
        verify:        If True, run LLM verification with retry loop.
    """
    print(f"\n[extract] Selecting section for: '{section_query}'")

    # Build unified candidate list: heading matches first, semantic fallbacks after
    candidate_docs: list[Document] = []
    seen_ids: set[int] = set()

    heading_candidates = select_heading_with_llm(
        index, section_query,
        llm_model=llm_model, base_url=base_url, api_key=api_key,
    )
    for doc in heading_candidates:
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            candidate_docs.append(doc)
            seen_ids.add(cid)

    if heading_candidates:
        print(f"      → Heading candidates: {[d.metadata.get('heading') for d in heading_candidates]}")
    else:
        print("      → No heading match; falling back to semantic search...")

    for doc, _ in retrieve_candidates(index, section_query):
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            candidate_docs.append(doc)
            seen_ids.add(cid)

    if not candidate_docs:
        raise ValueError(f"No candidates found for query: '{section_query}'")

    print(f"      → {len(candidate_docs)} total candidates")

    # Retry loop
    print("[extract] Gathering & verifying section...")
    result: ExtractionResult | None = None

    for attempt, candidate_doc in enumerate(candidate_docs, start=1):
        heading = candidate_doc.metadata.get("heading", "?")
        section_chunks = gather_full_section(candidate_doc, index.chunks)
        merged = merge_section_chunks(section_chunks)

        if not verify:
            result = merged
            print(f"      → Selected: {heading}")
            break

        verify_doc = Document(
            page_content=merged.content,
            metadata={"heading": merged.heading, "pages": merged.pages},
        )
        confirmed, llm_output = verify_with_llm(
            verify_doc, section_query,
            llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        result = pydantic_copy(merged, {"verified": confirmed, "llm_output": llm_output})

        if confirmed:
            print(f"      → Verified on attempt {attempt}: {heading}")
            break

        print(f"      → Attempt {attempt} failed ({heading}), trying next...")

    return result
