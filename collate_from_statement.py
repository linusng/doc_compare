"""
Multi-term collator — given a free-text statement, extracts every key
term/concept it references, retrieves relevant content for each term from
across the whole document, and returns a single collated context that spans
multiple sections.

Unlike extract_section_wllm.py (which finds ONE section), this script
collects fragments from MANY sections whose content collectively explains
the statement.

Usage:
    from collate_from_statement import collate_from_statement

    statement = (
        "Extension Option: At the Borrower's option, the Final Maturity Date "
        "can be extended up to two extensions of one year each from the "
        "Original Maturity Date."
    )
    result = collate_from_statement("facility_agreement.pdf", statement)
    print(result.combined_context)

CLI:
    python collate_from_statement.py \\
        --pdf facility_agreement.pdf \\
        --statement "Extension Option: ..."
"""

import argparse
import json
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from extract_multilingual_wllm import (
    _base_heading,
    build_vector_store,
    chunk_by_section,
    extract_structured_blocks,
    filter_toc_blocks,
    split_oversized_chunks,
)
from extract_section_wllm import _chunk_to_doc


# ── Output models ─────────────────────────────────────────────────────────────

class SectionEvidence(BaseModel):
    """Content found in one section relevant to the statement."""
    heading: str
    pages: list
    matched_terms: list       # which terms from the statement match here
    excerpt: str              # the section chunk text


class CollatedResult(BaseModel):
    """Full collation result across all matched sections."""
    statement: str
    terms_extracted: list     # key terms the LLM pulled from the statement
    match_count: int          # total unique chunks found
    sections_referenced: list # unique section headings, in page order
    evidence: list            # list[SectionEvidence]
    combined_context: str     # ready-to-use context string for downstream LLMs
    synthesis: str = ""       # optional LLM synthesis (if synthesize=True)


# ── Phase 5a: Term extraction ─────────────────────────────────────────────────

def extract_key_terms(
    statement: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[str]:
    """
    Ask the LLM to extract every key term, defined term, and concept from
    the statement that could appear in a legal agreement.

    Returns a deduplicated list of search terms, specific-to-broad:
      e.g. ["Final Maturity Date", "Extension Option", "Original Maturity Date",
             "Borrower", "Facility tenor", "utilisation date"]
    """
    llm = ChatOpenAI(
        model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst. Given a statement describing "
            "terms or clauses in a legal agreement, extract every key term, "
            "defined term, named mechanism, and concept that would appear "
            "in the agreement text.\n\n"
            "Extraction rules:\n"
            "- Capitalised defined terms: 'Final Maturity Date', 'Borrower'\n"
            "- Named mechanisms: 'Extension Option', 'Utilisation Request'\n"
            "- Section-level concepts: 'Repayment', 'Interest', 'Purpose'\n"
            "- Financial terms: 'Margin', 'Commitment Period', 'Facility'\n"
            "- Include both the exact phrase AND shorter variants if useful\n"
            "- Exclude generic words like 'the', 'option', 'date' alone\n\n"
            "Return ONLY a JSON array of strings, 5–15 terms, most specific first.\n"
            'Example: ["Final Maturity Date", "Extension Option", "Original Maturity Date", '
            '"Borrower", "Facility tenor"]'
        )),
        ("human", "{statement}"),
    ])

    result = (prompt | llm).invoke({"statement": statement})
    raw = result.content.strip()

    # Parse JSON array from the response
    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            terms = json.loads(m.group())
            return [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: treat comma/newline-separated lines as terms
    return [
        t.strip().strip('"').strip("'")
        for t in re.split(r'[,\n]', raw)
        if t.strip().strip('"').strip("'")
    ]


# ── Phase 5b: Per-term retrieval ──────────────────────────────────────────────

def retrieve_for_term(
    term: str,
    vector_store: InMemoryVectorStore,
    chunks: list,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Retrieve the most relevant chunks for a single term.

    Combines:
    - Semantic similarity search (embedding-based)
    - Heading keyword injection (exact heading match → score=1.0)

    Returns (doc, score) pairs, deduplicated by chunk_id.
    """
    seen: dict[int, tuple[Document, float]] = {}

    # Semantic search
    for doc, score in vector_store.similarity_search_with_score(term, k=top_k):
        cid = doc.metadata["chunk_id"]
        if cid not in seen or score > seen[cid][1]:
            seen[cid] = (doc, score)

    # Heading keyword match — guaranteed inclusion, max score
    term_lower = term.lower()
    for chunk in chunks:
        cid = chunk.chunk_id
        if term_lower in chunk.heading.lower() and cid not in seen:
            seen[cid] = (_chunk_to_doc(chunk), 1.0)

    return list(seen.values())


# ── Phase 6: Multi-term merge with boost ─────────────────────────────────────

def gather_all_relevant(
    terms: list[str],
    vector_store: InMemoryVectorStore,
    chunks: list,
    top_k_per_term: int = 5,
    score_threshold: float = 0.35,
) -> list[tuple[Document, float, list[str]]]:
    """
    Run per-term retrieval for every extracted term, then merge all results.

    Multi-term boost: a chunk matching N different terms has its score
    multiplied by sqrt(N).  This surfaces chunks that explain multiple
    concepts from the statement — e.g. a clause that defines both
    'Final Maturity Date' and 'Extension Option' ranks higher than one
    that only defines one.

    Returns a list of (doc, boosted_score, matched_terms) sorted by
    boosted_score descending.
    """
    # chunk_id → (doc, best_raw_score, [terms that matched])
    merged: dict[int, tuple[Document, float, list[str]]] = {}

    for term in terms:
        for doc, score in retrieve_for_term(term, vector_store, chunks, top_k=top_k_per_term):
            if score < score_threshold:
                continue
            cid = doc.metadata["chunk_id"]
            if cid not in merged:
                merged[cid] = (doc, score, [term])
            else:
                old_doc, old_score, old_terms = merged[cid]
                updated_terms = old_terms if term in old_terms else old_terms + [term]
                merged[cid] = (old_doc, max(old_score, score), updated_terms)

    # Apply multi-term boost and sort
    boosted = [
        (doc, score * (len(matched_terms) ** 0.5), matched_terms)
        for doc, score, matched_terms in merged.values()
    ]
    return sorted(boosted, key=lambda x: -x[1])


# ── Phase 7: Format collated context ─────────────────────────────────────────

def build_collated_context(
    ranked: list[tuple[Document, float, list[str]]],
    max_chunks: int = 25,
) -> tuple[str, list[SectionEvidence], list[str]]:
    """
    Group matched chunks by section heading, sort by page number, and
    render a structured context string.

    Returns:
        combined_context  — formatted string ready for downstream LLM use
        evidence          — list of SectionEvidence objects
        sections_referenced — unique headings in page order
    """
    # Group by base heading
    by_section: dict[str, list[tuple[Document, float, list[str]]]] = defaultdict(list)
    for doc, score, terms in ranked[:max_chunks]:
        heading = _base_heading(doc.metadata.get("heading", "(no heading)"))
        by_section[heading].append((doc, score, terms))

    # Sort sections by their earliest page
    def first_page(items):
        return min(
            (min(doc.metadata.get("pages", [9999])) for doc, _, _ in items),
            default=9999,
        )

    sorted_sections = sorted(by_section.items(), key=lambda kv: first_page(kv[1]))

    context_parts = []
    evidence_list = []
    sections_referenced = []

    for heading, items in sorted_sections:
        pages = sorted({p for doc, _, _ in items for p in doc.metadata.get("pages", [])})
        matched_terms = sorted({t for _, _, terms in items for t in terms})
        sections_referenced.append(heading)

        # Sort chunks within the section by start page
        items_sorted = sorted(items, key=lambda x: min(x[0].metadata.get("pages", [0])))

        # Deduplicate content within the section
        seen_texts: set[str] = set()
        excerpts = []
        for doc, _, _ in items_sorted:
            text = doc.page_content.strip()
            if text not in seen_texts:
                excerpts.append(text)
                seen_texts.add(text)

        excerpt_combined = "\n\n".join(excerpts)

        evidence_list.append(SectionEvidence(
            heading=heading,
            pages=pages,
            matched_terms=matched_terms,
            excerpt=excerpt_combined,
        ))

        # Format for combined_context
        context_parts.append(
            f"{'═' * 60}\n"
            f"SECTION : {heading}\n"
            f"Pages   : {pages}\n"
            f"Terms   : {', '.join(matched_terms)}\n"
            f"{'─' * 60}\n"
            f"{excerpt_combined}"
        )

    combined_context = "\n\n".join(context_parts)
    return combined_context, evidence_list, sections_referenced


# ── Phase 8 (optional): LLM synthesis ────────────────────────────────────────

def synthesize(
    statement: str,
    combined_context: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    """
    Given the original statement and all retrieved document context,
    ask the LLM to confirm, correct, or enrich the statement based on
    what the document actually says.
    """
    llm = ChatOpenAI(
        model=llm_model, base_url=base_url, api_key=api_key, temperature=0,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst. You will be given:\n"
            "1. A STATEMENT describing terms or clauses in a legal agreement.\n"
            "2. DOCUMENT CONTEXT — excerpts from the actual agreement, "
            "retrieved from multiple sections.\n\n"
            "Your task:\n"
            "- Verify whether the statement accurately reflects the document.\n"
            "- Identify any discrepancies, missing details, or nuances.\n"
            "- Produce a structured summary combining the statement with the "
            "evidence from the document, noting which sections support each point.\n\n"
            "Format your response as:\n"
            "VERIFIED POINTS: (what the statement gets right)\n"
            "DISCREPANCIES: (where the statement differs from the document)\n"
            "ADDITIONAL CONTEXT: (relevant detail from the document not in the statement)\n"
            "SOURCES: (section headings and pages that support the above)"
        )),
        ("human",
            "STATEMENT:\n{statement}\n\n"
            "DOCUMENT CONTEXT:\n{context}"
        ),
    ])

    result = (prompt | llm).invoke({
        "statement": statement,
        "context": combined_context[:6000],  # cap to avoid context overflow
    })
    return result.content.strip()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def collate_from_statement(
    pdf_path: str,
    statement: str,
    synthesize_result: bool = False,
    llm_model: str = "gemma3-27b-it",
    ollama_base_url: str = "http://localhost:11434/v1",
    ollama_api_key: str = "ollama",
    embedding_model: str = "bge-m3",
    top_k_per_term: int = 5,
    score_threshold: float = 0.35,
    max_chunks: int = 25,
) -> CollatedResult:
    """
    Full pipeline:

      PDF → blocks → filter TOC → chunk → split → embed
          → extract terms from statement
          → retrieve top-k chunks per term (semantic + heading keyword)
          → merge with multi-term boost
          → format collated context grouped by section
          → (optionally) LLM synthesis

    Args:
        pdf_path:         Path to the PDF file.
        statement:        Free-text description — may reference terms scattered
                          across many sections of the document.
        synthesize_result: If True, run a final LLM pass to verify and enrich
                          the statement against retrieved context.
        top_k_per_term:   How many chunks to retrieve per extracted term.
        score_threshold:  Minimum similarity score to include a chunk.
        max_chunks:       Cap on total chunks included in collated context.
    """
    print(f"[1/6] Extracting blocks from: {pdf_path}")
    blocks = extract_structured_blocks(pdf_path)
    print(f"      → {len(blocks)} blocks extracted")

    print("[2/6] Filtering TOC pages...")
    blocks = filter_toc_blocks(
        blocks,
        llm_model=llm_model,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
    )
    print(f"      → {len(blocks)} blocks after TOC filter")

    print("[3/6] Chunking & splitting...")
    chunks = chunk_by_section(blocks)
    chunks = split_oversized_chunks(chunks)
    print(f"      → {len(chunks)} chunks")

    print("[4/6] Building vector store...")
    vector_store = build_vector_store(
        chunks,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
        model=embedding_model,
    )

    print("[5/6] Extracting key terms from statement...")
    terms = extract_key_terms(
        statement,
        llm_model=llm_model,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
    )
    print(f"      → {len(terms)} terms: {terms}")

    print("[6/6] Retrieving & collating relevant chunks...")
    ranked = gather_all_relevant(
        terms,
        vector_store,
        chunks,
        top_k_per_term=top_k_per_term,
        score_threshold=score_threshold,
    )
    print(f"      → {len(ranked)} unique chunks matched")

    combined_context, evidence, sections_referenced = build_collated_context(
        ranked, max_chunks=max_chunks,
    )

    synthesis_text = ""
    if synthesize_result:
        print("      Running LLM synthesis...")
        synthesis_text = synthesize(
            statement,
            combined_context,
            llm_model=llm_model,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )

    return CollatedResult(
        statement=statement,
        terms_extracted=terms,
        match_count=len(ranked),
        sections_referenced=sections_referenced,
        evidence=evidence,
        combined_context=combined_context,
        synthesis=synthesis_text,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract and collate all document content relevant to a free-text "
            "statement, pulling from multiple sections."
        )
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF")
    parser.add_argument(
        "--statement", required=True,
        help="Free-text statement referencing terms scattered across the document",
    )
    parser.add_argument(
        "--synthesize", action="store_true",
        help="Run LLM synthesis pass after collation",
    )
    parser.add_argument("--llm-model",       default="gemma3-27b-it")
    parser.add_argument("--embedding-model", default="bge-m3")
    parser.add_argument("--base-url",        default="http://localhost:11434/v1")
    parser.add_argument(
        "--top-k",  type=int, default=5,
        help="Chunks retrieved per term (default: 5)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.35,
        help="Minimum similarity score to include a chunk (default: 0.35)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=25,
        help="Max chunks included in collated context (default: 25)",
    )
    args = parser.parse_args()

    result = collate_from_statement(
        pdf_path=args.pdf,
        statement=args.statement,
        synthesize_result=args.synthesize,
        llm_model=args.llm_model,
        ollama_base_url=args.base_url,
        ollama_api_key="ollama",
        embedding_model=args.embedding_model,
        top_k_per_term=args.top_k,
        score_threshold=args.threshold,
        max_chunks=args.max_chunks,
    )

    print("\n── Collation Result ───────────────────────────────────────")
    print(f"Terms extracted  : {result.terms_extracted}")
    print(f"Chunks matched   : {result.match_count}")
    print(f"Sections touched : {result.sections_referenced}")
    print(f"\n{result.combined_context}")

    if result.synthesis:
        print("\n── LLM Synthesis ──────────────────────────────────────────")
        print(result.synthesis)
