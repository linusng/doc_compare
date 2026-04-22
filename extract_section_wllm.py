"""
Generic section extractor — finds ANY named section in a PDF.

Usage:
    from extract_section_wllm import extract_section

    result = extract_section(
        pdf_path="facility_agreement.pdf",
        section_query="3.1 Purpose",   # or "Representations and Warranties", etc.
        verify=False,
    )
    print(result.heading, result.pages)

CLI:
    python extract_section_wllm.py --pdf facility_agreement.pdf --section "3.1 Purpose"
"""

import argparse
import re

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI

# ── Re-use all infrastructure from the definitions extractor ──────────────────
from extract_multilingual_wllm import (
    ExtractionResult,
    _base_heading,
    _is_child_section,
    build_vector_store,
    chunk_by_section,
    extract_structured_blocks,
    filter_toc_blocks,
    gather_full_section,
    merge_section_chunks,
    pydantic_copy,
    split_oversized_chunks,
)


# ── Phase 5: Heading-first Section Selection ──────────────────────────────────

def _chunk_to_doc(chunk) -> Document:
    return Document(
        page_content=chunk.full_text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "heading": chunk.heading,
            "pages": chunk.pages,
            "start_page": chunk.start_page,
            "content_length": len(chunk.content),
            "heading_level": chunk.heading_level,
        },
    )


def select_heading_with_llm(
    chunks: list,
    section_query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    top_n: int = 3,
) -> list[Document]:
    """
    Present every unique section heading to the LLM and ask it to rank
    up to top_n matches for section_query, best first.

    Returns a list of Documents in ranked order (may be empty if the LLM
    finds no matches).  Returning multiple candidates lets the caller retry
    with lower-ranked options when verification fails.
    """
    seen_bases: dict[str, object] = {}
    for chunk in chunks:
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

    llm = ChatOpenAI(
        model=llm_model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )

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
            f"e.g.: 5,2,8\n"
            "If no heading matches, respond with: NONE"
        )),
        ("human", "{headings}"),
    ])

    result = (prompt | llm).invoke({"headings": headings_text})
    answer = result.content.strip()

    if answer.upper() == "NONE":
        return []

    docs = []
    seen_ids: set[int] = set()
    for m in re.finditer(r'\d+', answer):
        idx = int(m.group())
        if 0 <= idx < len(unique_chunks) and idx not in seen_ids:
            docs.append(_chunk_to_doc(unique_chunks[idx]))
            seen_ids.add(idx)
        if len(docs) >= top_n:
            break

    return docs


def retrieve_candidates(
    vector_store: InMemoryVectorStore,
    section_query: str,
    chunks: list,
    top_k: int = 8,
) -> list[tuple[Document, float]]:
    """
    Fallback hybrid search used when heading selection fails.
    Runs multiple query angles through the vector store and also
    injects any chunks whose heading contains the query text at
    maximum score so they always appear at the top.
    """
    seen: dict[int, tuple[Document, float]] = {}

    queries = [
        section_query,
        f"Section {section_query}",
        f"{section_query} clause provisions",
    ]
    for query in queries:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        for doc, score in results:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen or score > seen[chunk_id][1]:
                seen[chunk_id] = (doc, score)

    query_lower = section_query.lower()
    for chunk in chunks:
        if query_lower in chunk.heading.lower() and chunk.chunk_id not in seen:
            seen[chunk.chunk_id] = (_chunk_to_doc(chunk), 1.0)

    return sorted(seen.values(), key=lambda x: -x[1])


# ── Phase 5b: LLM Ranking ─────────────────────────────────────────────────────

def pick_best_with_llm(
    candidates: list[tuple[Document, float]],
    section_query: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> Document:
    """
    Present top semantic-search candidates to the LLM and ask it to pick
    the one that best matches section_query.

    Falls back to the highest-scoring embedding candidate if the LLM
    can't decide or returns an invalid choice.
    """
    top = candidates[:8]

    options = []
    for i, (doc, sim_score) in enumerate(top):
        heading = doc.metadata.get("heading", "(no heading)")
        preview = doc.page_content[:300].replace("\n", " ")
        options.append(f"[{i}] Heading: {heading}\n    Preview: {preview}...")

    options_text = "\n\n".join(options)

    llm = ChatOpenAI(
        model=llm_model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst. You will be shown candidate "
            "sections from a legal agreement. Your task is to identify which "
            "candidate best matches the following section description:\n\n"
            f"  \"{section_query}\"\n\n"
            "Consider both the heading text and the content preview when "
            "deciding. The section number in the query (if any) is a hint, "
            "not an exact requirement — match by meaning if the numbering "
            "differs.\n\n"
            "Respond with ONLY the number in square brackets, e.g.: 0\n"
            "If none of the candidates matches, respond with: NONE"
        )),
        ("human", "{options}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"options": options_text})
    answer = result.content.strip()

    m = re.search(r'\d+', answer)
    if m:
        idx = int(m.group())
        if 0 <= idx < len(top):
            return top[idx][0]

    return top[0][0]


# ── Phase 6: LLM Verification ─────────────────────────────────────────────────

def verify_with_llm(
    doc: Document,
    section_query: str,
    model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> tuple:
    """
    Ask the LLM to confirm the retrieved chunk matches section_query.
    Returns (is_confirmed, cleaned_text).
    """
    llm = ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=0,
    )

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

    chain = prompt | llm
    result = chain.invoke({"text": doc.page_content[:4000]})
    output = result.content.strip()
    return output != "NOT_FOUND", output


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def extract_section(
    pdf_path: str,
    section_query: str,
    verify: bool = False,
    llm_model: str = "gemma3-27b-it",
    ollama_base_url: str = "http://localhost:11434/v1",
    ollama_api_key: str = "ollama",
    embedding_model: str = "bge-m3",
) -> ExtractionResult:
    """
    Full pipeline:
      PDF → structured blocks → filter TOC pages → section chunks
          → token-safe splits → embeddings → semantic retrieval
          → LLM ranking → gather full section → merge → ExtractionResult

    Args:
        pdf_path:      Path to the PDF file.
        section_query: Free-text description of the section to find,
                       e.g. "3.1 Purpose" or "Representations and Warranties".
    """
    print(f"[1/7] Extracting blocks from: {pdf_path}")
    blocks = extract_structured_blocks(pdf_path)
    print(f"      → {len(blocks)} blocks extracted")

    print("[2/7] Filtering TOC pages...")
    blocks = filter_toc_blocks(
        blocks,
        llm_model=llm_model,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
    )
    print(f"      → {len(blocks)} blocks after TOC filtering")

    print("[3/7] Chunking by section...")
    chunks = chunk_by_section(blocks)
    print(f"      → {len(chunks)} sections found")

    print("[4/7] Splitting oversized chunks...")
    chunks = split_oversized_chunks(chunks)
    print(f"      → {len(chunks)} chunks after token-safe split")

    print("[5/7] Building vector store...")
    vector_store = build_vector_store(
        chunks,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
        model=embedding_model,
    )

    print(f"[6/7] Selecting section for: '{section_query}'...")
    # Build an ordered candidate list: heading-ranked matches first,
    # then semantic fallbacks.  The retry loop below works through this
    # list so verification failures automatically try the next candidate.
    candidate_docs: list[Document] = []
    seen_ids: set[int] = set()

    heading_candidates = select_heading_with_llm(
        chunks,
        section_query=section_query,
        llm_model=llm_model,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
    )
    for doc in heading_candidates:
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            candidate_docs.append(doc)
            seen_ids.add(cid)

    if heading_candidates:
        labels = [d.metadata.get("heading", "?") for d in heading_candidates]
        print(f"      → Heading candidates: {labels}")
    else:
        print("      → No heading match; using semantic search only...")

    # Always append semantic candidates as additional fallbacks.
    sem_candidates = retrieve_candidates(vector_store, section_query, chunks)
    for doc, _ in sem_candidates:
        cid = doc.metadata["chunk_id"]
        if cid not in seen_ids:
            candidate_docs.append(doc)
            seen_ids.add(cid)

    if not candidate_docs:
        raise ValueError(f"No candidates found for query: '{section_query}'")

    print(f"      → {len(candidate_docs)} total candidates to try")

    print("[7/7] Gathering & verifying section...")
    result = None
    for attempt, candidate_doc in enumerate(candidate_docs, start=1):
        heading = candidate_doc.metadata.get("heading", "?")
        section_chunks = gather_full_section(candidate_doc, chunks)
        merged = merge_section_chunks(section_chunks)

        if not verify:
            # No verification requested — take the top candidate immediately.
            result = merged
            print(f"      → Selected: {heading}")
            break

        verify_doc = Document(
            page_content=merged.content,
            metadata={"heading": merged.heading, "pages": merged.pages},
        )
        confirmed, llm_output = verify_with_llm(
            verify_doc,
            section_query=section_query,
            model=llm_model,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )
        result = pydantic_copy(merged, {"verified": confirmed, "llm_output": llm_output})

        if confirmed:
            print(f"      → Verified on attempt {attempt}: {heading}")
            break

        print(f"      → Attempt {attempt} failed verification ({heading}), trying next...")

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract any named section from a PDF using LLM-assisted semantic search."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file")
    parser.add_argument("--section", required=True, help='Section to find, e.g. "3.1 Purpose"')
    parser.add_argument("--verify", action="store_true", help="Run LLM verification step")
    parser.add_argument("--llm-model", default="gemma3-27b-it")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    args = parser.parse_args()

    result = extract_section(
        pdf_path=args.pdf,
        section_query=args.section,
        verify=args.verify,
        llm_model=args.llm_model,
        ollama_base_url=args.base_url,
        ollama_api_key="ollama",
        embedding_model=args.embedding_model,
    )

    print("\n── Extraction Result ──────────────────────────────────────")
    print(f"Section  : {result.heading}")
    print(f"Pages    : {result.pages}")
    print(f"Verified : {result.verified}")
    print(f"\n{result.content[:800]}")
