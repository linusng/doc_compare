"""
Steps 1–5: PDF → text blocks → TOC filter → section chunks
           → token-safe splits → vector store.

The DocumentIndex dataclass holds the built artefacts so that multiple
extract/collate calls against the same PDF share one vector store without
ever rebuilding it.

Typical usage:

    from pipeline.ingestion import DocumentIndex

    index = DocumentIndex.from_pdf("agreement.pdf")

    # Pass `index` to run_extract() or run_collate() as many times as needed.
"""

import re
import statistics
from dataclasses import dataclass

import pymupdf
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from .models import SectionChunk, TextBlock, pydantic_copy


# ── Tokeniser (BGE-M3) ────────────────────────────────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
BGE_M3_MAX_TOKENS = 8192
BGE_M3_CHUNK_OVERLAP = 50


def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=True))


# ── Step 1: Extract structured blocks ─────────────────────────────────────────

def extract_structured_blocks(pdf_path: str) -> list[TextBlock]:
    """
    Extract text blocks from a PDF, retaining per-block font metadata.
    Font size and bold flag drive heading detection in the next step.
    """
    doc = pymupdf.open(pdf_path)
    blocks: list[TextBlock] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue  # skip image blocks

            full_text = ""
            max_font_size = 0.0
            is_bold = False

            for line in block["lines"]:
                for span in line["spans"]:
                    full_text += span["text"]
                    max_font_size = max(max_font_size, span["size"])
                    if "bold" in span["font"].lower():
                        is_bold = True
                full_text += "\n"

            if full_text.strip():
                blocks.append(TextBlock(
                    text=full_text.strip(),
                    page=page_num,
                    bbox=tuple(block["bbox"]),
                    font_size=max_font_size,
                    is_bold=is_bold,
                    block_no=block["number"],
                ))

    doc.close()
    return blocks


# ── Step 2: TOC page detection and filtering ───────────────────────────────────

def detect_toc_pages(
    blocks: list[TextBlock],
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    max_pages_to_check: int = 15,
) -> set[int]:
    """
    Use an LLM to identify Table of Contents pages in the first N pages.

    Only pages we actually showed the LLM can be returned as TOC pages.
    A sanity check rejects results with more than 5 pages or non-consecutive
    pages, both of which are signs of a hallucinated response.
    """
    pages: dict[int, list[TextBlock]] = {}
    for b in blocks:
        pages.setdefault(b.page, []).append(b)

    pages_to_check = sorted(pages.keys())[:max_pages_to_check]
    if not pages_to_check:
        return set()

    page_previews = []
    for page_num in pages_to_check:
        page_text = "\n".join(b.text for b in pages[page_num])
        page_previews.append(f"--- PAGE {page_num} ---\n{page_text[:500]}")

    pages_text = "\n\n".join(page_previews)

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a document structure analyst. You will be shown the "
            "text content of several pages from a legal document.\n\n"
            "Your task is to identify which pages are a Table of Contents "
            "(TOC). A TOC page contains ONLY:\n"
            "- Lists of section numbers and titles with page number references\n"
            "- Dot leaders or spacing between titles and page numbers\n"
            "- Dense listings of headings with NO substantive body text\n\n"
            "A TOC is NOT:\n"
            "- A cover page, title page, or signature page\n"
            "- A page with actual clause or contract body text\n"
            "- A page that merely starts with a section heading\n\n"
            "Note: The text extraction may be messy — section titles and page "
            "numbers may run together without spaces (e.g. "
            "'1.DEFINITIONS AND INTERPRETATION2 2.THE FACILITY13'). "
            "This is still a TOC.\n\n"
            "Be conservative — only include a page if you are CERTAIN it is "
            "a TOC page. When in doubt, exclude it.\n\n"
            "Respond with ONLY a comma-separated list of page numbers that "
            "are TOC pages, using the exact PAGE numbers shown in the input. "
            "If no pages are TOC pages, respond with: NONE\n\n"
            "Example response: 2,3"
        )),
        ("human", "{pages}"),
    ])

    answer = (prompt | llm).invoke({"pages": pages_text}).content.strip()

    if answer.upper() == "NONE":
        return set()

    shown_pages = set(pages_to_check)
    toc_pages: set[int] = set()
    for token in answer.replace(" ", "").split(","):
        try:
            page_num = int(token)
            if page_num in shown_pages:
                toc_pages.add(page_num)
        except ValueError:
            continue

    # Sanity checks: legal TOC is always a small, consecutive block at the start.
    MAX_TOC_PAGES = 5
    if len(toc_pages) > MAX_TOC_PAGES:
        return set()
    if len(toc_pages) > 1:
        sorted_toc = sorted(toc_pages)
        if sorted_toc[-1] - sorted_toc[0] > MAX_TOC_PAGES:
            return set()

    return toc_pages


def filter_toc_blocks(
    blocks: list[TextBlock],
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> list[TextBlock]:
    """Remove all blocks that belong to detected TOC pages."""
    toc_pages = detect_toc_pages(
        blocks, llm_model=llm_model, base_url=base_url, api_key=api_key,
    )
    if toc_pages:
        print(f"      → TOC detected on page(s): {sorted(toc_pages)}, excluding")
    else:
        print("      → No TOC pages detected")
    return [b for b in blocks if b.page not in toc_pages]


# ── Step 3: Heading detection and section chunking ────────────────────────────

def _get_heading_level(text: str) -> int:
    """
    Infer heading depth from the section number prefix.
    '1.' → 1,  '1.1' → 2,  '1.1.1' → 3,  no number → 0.
    """
    m = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s', text.strip())
    if not m:
        return 0
    return len([p for p in m.group(1).split(".") if p])


def detect_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Mark blocks as headings based on font size and boldness.
    Language-agnostic — works on any script.

    A block is a heading if:
    - Its font size exceeds 1.1× the document median, OR
    - It is bold and short (< 120 characters).
    """
    font_sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not font_sizes:
        return blocks

    median_size = statistics.median(font_sizes)
    updated = []
    for block in blocks:
        is_heading = (
            block.font_size > median_size * 1.1
            or (block.is_bold and len(block.text) < 120)
        )
        updated.append(pydantic_copy(block, {"is_heading": is_heading}))
    return updated


def chunk_by_section(blocks: list[TextBlock]) -> list[SectionChunk]:
    """
    Produce one SectionChunk per section: heading + all body text beneath it.
    Page metadata is accumulated across all blocks in the section.
    """
    blocks = detect_headings(blocks)
    chunks: list[SectionChunk] = []
    current_heading = ""
    current_body: list[str] = []
    current_pages: set[int] = set()

    def flush(chunk_id: int) -> SectionChunk | None:
        if not current_heading and not current_body:
            return None
        return SectionChunk(
            heading=current_heading,
            content="\n".join(current_body),
            pages=sorted(current_pages),
            start_page=min(current_pages) if current_pages else 0,
            chunk_id=chunk_id,
            heading_level=_get_heading_level(current_heading),
        )

    for block in blocks:
        if block.is_heading:
            saved = flush(len(chunks))
            if saved:
                chunks.append(saved)
            current_heading = block.text
            current_body = []
            current_pages = {block.page}
        else:
            current_body.append(block.text)
            current_pages.add(block.page)

    saved = flush(len(chunks))
    if saved:
        chunks.append(saved)

    return chunks


# ── Step 4: Token-safe splitting ──────────────────────────────────────────────

def split_oversized_chunks(
    chunks: list[SectionChunk],
    max_tokens: int = BGE_M3_MAX_TOKENS,
    chunk_overlap: int = BGE_M3_CHUNK_OVERLAP,
) -> list[SectionChunk]:
    """
    Any SectionChunk whose full_text exceeds BGE-M3's 8,192-token limit is
    split into sub-chunks using the real BGE-M3 tokeniser as the length
    function. Heading and page metadata are propagated to every sub-chunk.
    Splits are labelled "(part 1)", "(part 2)", etc.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result: list[SectionChunk] = []
    for chunk in chunks:
        if token_len(chunk.full_text) <= max_tokens:
            result.append(chunk)
            continue

        for i, sub_text in enumerate(splitter.split_text(chunk.content)):
            result.append(SectionChunk(
                heading=f"{chunk.heading} (part {i + 1})",
                content=sub_text,
                pages=chunk.pages,
                start_page=chunk.start_page,
                chunk_id=0,
                heading_level=chunk.heading_level,
            ))

    # Re-assign chunk_ids sequentially after splits
    for idx, chunk in enumerate(result):
        result[idx] = pydantic_copy(chunk, {"chunk_id": idx})

    return result


# ── Step 5: Vector store ──────────────────────────────────────────────────────

def build_vector_store(
    chunks: list[SectionChunk],
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model: str = "bge-m3",
) -> InMemoryVectorStore:
    """
    Embed all section chunks with BGE-M3 and load into an in-memory vector
    store. Heading and body text are embedded together so heading signals
    contribute to chunk representations.
    """
    embeddings = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)

    docs = [
        Document(
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
        for chunk in chunks
    ]

    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    return store


# ── DocumentIndex: built once, reused many times ──────────────────────────────

@dataclass
class DocumentIndex:
    """
    Pre-built artefacts for one PDF document.

    Build once with DocumentIndex.from_pdf(), then pass to run_extract()
    or run_collate() as many times as needed — the vector store is never
    rebuilt between calls.

    Example
    -------
    >>> from pipeline.ingestion import DocumentIndex
    >>> from pipeline.extract import run_extract
    >>> from pipeline.collate import run_collate
    >>>
    >>> index = DocumentIndex.from_pdf("agreement.pdf")
    >>>
    >>> r1 = run_extract(index, "Schedule 2 Compounded Rate Terms")
    >>> r2 = run_extract(index, "3.1 Purpose")
    >>> r3 = run_collate(index, "Extension Option: the Final Maturity Date...")
    """
    pdf_path: str
    chunks: list[SectionChunk]
    vector_store: InMemoryVectorStore

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        llm_model: str = "gemma3-27b-it",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        embedding_model: str = "bge-m3",
    ) -> "DocumentIndex":
        """
        Run the five shared preparation phases and return a DocumentIndex.
        This is the only step that reads the PDF — all subsequent operations
        work entirely from the returned index.
        """
        print(f"[1/5] Extracting blocks from: {pdf_path}")
        blocks = extract_structured_blocks(pdf_path)
        print(f"      → {len(blocks)} blocks extracted")

        print("[2/5] Filtering TOC pages...")
        blocks = filter_toc_blocks(
            blocks, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        print(f"      → {len(blocks)} blocks after TOC filter")

        print("[3/5] Chunking by section...")
        chunks = chunk_by_section(blocks)
        print(f"      → {len(chunks)} sections found")

        print("[4/5] Splitting oversized chunks...")
        chunks = split_oversized_chunks(chunks)
        print(f"      → {len(chunks)} chunks after token-safe split")

        print("[5/5] Building vector store...")
        vector_store = build_vector_store(
            chunks, base_url=base_url, api_key=api_key, model=embedding_model,
        )
        print(f"      → Vector store ready ({len(chunks)} documents indexed)")

        return cls(pdf_path=pdf_path, chunks=chunks, vector_store=vector_store)
