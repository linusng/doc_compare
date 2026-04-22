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
    split_oversized_chunks,
)


# ── Phase 5: Generic Semantic Retrieval ───────────────────────────────────────

def retrieve_candidates(
    vector_store: InMemoryVectorStore,
    section_query: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Search the vector store using the caller-supplied query string.
    Returns (doc, best_similarity_score) tuples deduplicated and sorted
    by score descending.
    """
    seen: dict[int, tuple[Document, float]] = {}

    for query in [section_query]:
        results = vector_store.similarity_search_with_score(query, k=top_k)
        for doc, score in results:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen or score > seen[chunk_id][1]:
                seen[chunk_id] = (doc, score)

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

    print(f"[6/7] Retrieving & ranking candidates for: '{section_query}'...")
    candidates = retrieve_candidates(vector_store, section_query)
    print(f"      → {len(candidates)} unique candidates from semantic search")
    best = pick_best_with_llm(
        candidates,
        section_query=section_query,
        llm_model=llm_model,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
    )
    print(f"      → LLM selected: {best.metadata.get('heading', '?')}")

    print("[7/7] Gathering full section...")
    section_chunks = gather_full_section(best, chunks)
    print(f"      → {len(section_chunks)} chunks in full section")
    result = merge_section_chunks(section_chunks)

    if verify:
        print("      Running LLM verification...")
        verify_doc = Document(
            page_content=result.content,
            metadata={"heading": result.heading, "pages": result.pages},
        )
        confirmed, llm_output = verify_with_llm(
            verify_doc,
            section_query=section_query,
            model=llm_model,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )
        from extract_multilingual_wllm import pydantic_copy
        result = pydantic_copy(result, {
            "verified": confirmed,
            "llm_output": llm_output,
        })

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
