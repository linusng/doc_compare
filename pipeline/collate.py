"""
Collate mode: given a free-text statement, extract key terms, retrieve
evidence from across the document, and return a single unified context.

Main entry point: run_collate(index, statement, ...)
"""

import json
import re
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .ingestion import DocumentIndex
from .models import CollatedResult, SectionEvidence
from .section_utils import _base_heading, chunk_to_doc


# ── Term extraction ───────────────────────────────────────────────────────────

def extract_key_terms(
    statement: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[str]:
    """
    Ask the LLM to extract every key term, defined term, and concept from
    the statement that could appear in a legal agreement.

    Returns a deduplicated list of search terms, most specific first:
      e.g. ["Final Maturity Date", "Extension Option", "Original Maturity Date",
             "Borrower", "Facility tenor"]
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

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
            'Example: ["Final Maturity Date", "Extension Option", '
            '"Original Maturity Date", "Borrower", "Facility tenor"]'
        )),
        ("human", "{statement}"),
    ])

    raw = (prompt | llm).invoke({"statement": statement}).content.strip()

    m = re.search(r'\[.*?\]', raw, re.DOTALL)
    if m:
        try:
            terms = json.loads(m.group())
            return [t.strip() for t in terms if isinstance(t, str) and t.strip()]
        except json.JSONDecodeError:
            pass

    # Fallback: split on commas/newlines and strip quotes
    return [
        t.strip().strip('"').strip("'")
        for t in re.split(r'[,\n]', raw)
        if t.strip().strip('"').strip("'")
    ]


# ── Per-term retrieval ────────────────────────────────────────────────────────

def retrieve_for_term(
    term: str,
    index: DocumentIndex,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Retrieve the most relevant chunks for a single term.

    Combines:
    - Semantic similarity search (embedding-based)
    - Heading keyword injection (exact heading match → score=1.0)

    Returns (doc, score) pairs, deduplicated by chunk_id, highest score wins.
    """
    seen: dict[int, tuple[Document, float]] = {}

    for doc, score in index.vector_store.similarity_search_with_score(term, k=top_k):
        cid = doc.metadata["chunk_id"]
        if cid not in seen or score > seen[cid][1]:
            seen[cid] = (doc, score)

    term_lower = term.lower()
    for chunk in index.chunks:
        if term_lower in chunk.heading.lower() and chunk.chunk_id not in seen:
            seen[chunk.chunk_id] = (chunk_to_doc(chunk), 1.0)

    return list(seen.values())


# ── Multi-term merge with boost ───────────────────────────────────────────────

def gather_all_relevant(
    terms: list[str],
    index: DocumentIndex,
    top_k_per_term: int = 5,
    score_threshold: float = 0.35,
) -> list[tuple[Document, float, list[str]]]:
    """
    Run per-term retrieval for every extracted term, merge results, and apply
    a multi-term boost.

    Multi-term boost: a chunk matching N different terms has its score
    multiplied by sqrt(N). This surfaces chunks that explain multiple concepts
    from the statement — e.g. a clause that defines both 'Final Maturity Date'
    and 'Extension Option' ranks higher than one that only defines one.

    Returns a list of (doc, boosted_score, matched_terms) sorted by
    boosted_score descending.
    """
    # chunk_id → (doc, best_raw_score, [matched terms])
    merged: dict[int, tuple[Document, float, list[str]]] = {}

    for term in terms:
        for doc, score in retrieve_for_term(term, index, top_k=top_k_per_term):
            if score < score_threshold:
                continue
            cid = doc.metadata["chunk_id"]
            if cid not in merged:
                merged[cid] = (doc, score, [term])
            else:
                old_doc, old_score, old_terms = merged[cid]
                updated_terms = old_terms if term in old_terms else old_terms + [term]
                merged[cid] = (old_doc, max(old_score, score), updated_terms)

    boosted = [
        (doc, score * (len(matched_terms) ** 0.5), matched_terms)
        for doc, score, matched_terms in merged.values()
    ]
    return sorted(boosted, key=lambda x: -x[1])


# ── Context assembly ──────────────────────────────────────────────────────────

def build_collated_context(
    ranked: list[tuple[Document, float, list[str]]],
    max_chunks: int = 25,
) -> tuple[str, list[SectionEvidence], list[str]]:
    """
    Group matched chunks by section heading, sort by page number, and
    render a structured context string ready for downstream LLM use.

    Returns:
        combined_context      — formatted string
        evidence              — list[SectionEvidence]
        sections_referenced   — unique headings in page order
    """
    by_section: dict[str, list] = defaultdict(list)
    for doc, score, terms in ranked[:max_chunks]:
        heading = _base_heading(doc.metadata.get("heading", "(no heading)"))
        by_section[heading].append((doc, score, terms))

    def first_page(items):
        return min(
            (min(doc.metadata.get("pages", [9999])) for doc, _, _ in items),
            default=9999,
        )

    sorted_sections = sorted(by_section.items(), key=lambda kv: first_page(kv[1]))

    context_parts: list[str] = []
    evidence_list: list[SectionEvidence] = []
    sections_referenced: list[str] = []

    for heading, items in sorted_sections:
        pages = sorted({p for doc, _, _ in items for p in doc.metadata.get("pages", [])})
        matched_terms = sorted({t for _, _, terms in items for t in terms})
        sections_referenced.append(heading)

        items_sorted = sorted(items, key=lambda x: min(x[0].metadata.get("pages", [0])))

        seen_texts: set[str] = set()
        excerpts: list[str] = []
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

        context_parts.append(
            f"{'═' * 60}\n"
            f"SECTION : {heading}\n"
            f"Pages   : {pages}\n"
            f"Terms   : {', '.join(matched_terms)}\n"
            f"{'─' * 60}\n"
            f"{excerpt_combined}"
        )

    return "\n\n".join(context_parts), evidence_list, sections_referenced


# ── Optional synthesis pass ───────────────────────────────────────────────────

def synthesize(
    statement: str,
    combined_context: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> str:
    """
    Compare the original statement against the retrieved document context.
    Returns a structured analysis with verified points, discrepancies,
    additional context, and source attributions.
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

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

    return (prompt | llm).invoke({
        "statement": statement,
        "context": combined_context[:6000],
    }).content.strip()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_collate(
    index: DocumentIndex,
    statement: str,
    synthesize_result: bool = False,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    top_k_per_term: int = 5,
    score_threshold: float = 0.35,
    max_chunks: int = 25,
) -> CollatedResult:
    """
    Collate evidence for a free-text statement from a pre-built DocumentIndex.

    1. LLM extracts key terms from the statement.
    2. Per-term retrieval: semantic search + heading keyword injection.
    3. Multi-term boost (score × √N) + threshold filter.
    4. Group by section, sort by page, build structured context.
    5. Optionally, run LLM synthesis to verify and enrich the statement.

    Args:
        index:             Pre-built DocumentIndex (call DocumentIndex.from_pdf first).
        statement:         Free-text description referencing terms scattered
                           across multiple sections.
        synthesize_result: If True, run a final LLM synthesis pass.
        top_k_per_term:    Chunks retrieved per extracted term (default 5).
        score_threshold:   Min similarity score to include a chunk (default 0.35).
        max_chunks:        Cap on chunks in the collated context (default 25).
    """
    print(f"\n[collate] Extracting key terms from statement...")
    terms = extract_key_terms(
        statement, llm_model=llm_model, base_url=base_url, api_key=api_key,
    )
    print(f"      → {len(terms)} terms: {terms}")

    print("[collate] Retrieving & collating relevant chunks...")
    ranked = gather_all_relevant(
        terms, index,
        top_k_per_term=top_k_per_term,
        score_threshold=score_threshold,
    )
    print(f"      → {len(ranked)} unique chunks matched")

    combined_context, evidence, sections_referenced = build_collated_context(
        ranked, max_chunks=max_chunks,
    )

    synthesis_text = ""
    if synthesize_result:
        print("[collate] Running LLM synthesis...")
        synthesis_text = synthesize(
            statement, combined_context,
            llm_model=llm_model, base_url=base_url, api_key=api_key,
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
