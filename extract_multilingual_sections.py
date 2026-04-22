import pymupdf
import statistics
import re
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# ── Config ────────────────────────────────────────────────────────────────────

BGE_M3_MAX_TOKENS = 8192
BGE_M3_CHUNK_OVERLAP = 50
_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")


# ── Models ────────────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    text: str
    page: int
    bbox: tuple
    font_size: float
    is_bold: bool
    block_no: int
    is_heading: bool = False


class SectionChunk(BaseModel):
    heading: str
    content: str
    pages: list[int]
    start_page: int
    chunk_id: int = 0
    parent_heading: str = ""   # original heading before "(part N)" suffix

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class SectionResult(BaseModel):
    query: str
    heading: str
    content: str
    pages: list[int]
    chunk_ids: list[int]        # all chunk_ids that were merged to form this result
    verified: bool = False
    llm_output: str | None = None


# ── Phase 1: Structured Extraction ───────────────────────────────────────────

def extract_structured_blocks(pdf_path: str) -> list[TextBlock]:
    """Extract text blocks from PDF, preserving font metadata for heading detection."""
    doc = pymupdf.open(pdf_path)
    blocks: list[TextBlock] = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

        for block in page_dict["blocks"]:
            if block["type"] != 0:
                continue

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


# ── Phase 2: Heading Detection & Section Chunking ────────────────────────────

def detect_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Mark blocks as headings based on font size deviation and boldness.
    Language-agnostic — works on any script.
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
        updated.append(block.copy(update={"is_heading": is_heading}))

    return updated


def chunk_by_section(blocks: list[TextBlock]) -> list[SectionChunk]:
    """
    Produce one SectionChunk per section: heading + all body text beneath it.
    Preserves page metadata throughout.
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
            parent_heading=current_heading,
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


# ── Phase 3: Token-safe Splitting ────────────────────────────────────────────

def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=True))


def split_oversized_chunks(
    chunks: list[SectionChunk],
    max_tokens: int = BGE_M3_MAX_TOKENS,
    chunk_overlap: int = BGE_M3_CHUNK_OVERLAP,
) -> list[SectionChunk]:
    """
    Split any SectionChunk that exceeds BGE-M3's token limit.
    Sub-chunks carry the parent_heading so they can be reassembled later.
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

        sub_texts = splitter.split_text(chunk.content)
        for i, sub_text in enumerate(sub_texts):
            result.append(SectionChunk(
                heading=f"{chunk.heading} (part {i + 1})",
                content=sub_text,
                pages=chunk.pages,
                start_page=chunk.start_page,
                chunk_id=0,
                parent_heading=chunk.heading,   # preserve original for reassembly
            ))

    # Re-assign chunk_ids sequentially after splits
    return [c.copy(update={"chunk_id": idx}) for idx, c in enumerate(result)]


# ── Phase 4: Embed into InMemoryVectorStore ───────────────────────────────────

def build_vector_store(
    chunks: list[SectionChunk],
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model: str = "bge-m3",
) -> tuple[InMemoryVectorStore, list[SectionChunk]]:
    """
    Embed all section chunks using BGE-M3 and load into InMemoryVectorStore.
    Returns both the store and the final chunk list (needed for reassembly).
    """
    embeddings = OpenAIEmbeddings(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )

    docs = [
        Document(
            page_content=chunk.full_text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading,
                "parent_heading": chunk.parent_heading,
                "pages": chunk.pages,
                "start_page": chunk.start_page,
                "content_length": len(chunk.content),
            },
        )
        for chunk in chunks
    ]

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(docs)
    return vector_store, chunks


# ── Phase 5: Section Search & Full Reassembly ────────────────────────────────

def search_section(
    query: str,
    vector_store: InMemoryVectorStore,
    all_chunks: list[SectionChunk],
    top_k: int = 3,
) -> SectionResult:
    """
    Given a section name query, finds the best matching section heading
    and returns ALL content from that heading until the next same-level
    heading — guaranteeing the entire section is returned.

    Strategy:
      1. Semantic search to identify the best matching chunk.
      2. Walk all_chunks from that chunk's position forward.
      3. Stop only when a new heading of equal/higher level is encountered.
      4. Concatenate everything in between.
    """
    results = vector_store.similarity_search(query, k=top_k)

    if not results:
        return SectionResult(
            query=query,
            heading="",
            content="",
            pages=[],
            chunk_ids=[],
        )

    # ── Step 1: resolve the best matching parent heading ─────────────────────
    target_parent_heading = results[0].metadata["parent_heading"]

    # Find the first chunk in all_chunks whose parent_heading matches
    start_idx = next(
        (i for i, c in enumerate(all_chunks)
         if c.parent_heading == target_parent_heading),
        None,
    )

    if start_idx is None:
        return SectionResult(
            query=query,
            heading="",
            content="",
            pages=[],
            chunk_ids=[],
        )

    # ── Step 2: detect the heading level of the matched section ──────────────
    matched_heading = all_chunks[start_idx].parent_heading
    matched_level = _heading_level(matched_heading)

    # ── Step 3: walk forward, collecting chunks until next same/higher heading
    collected: list[SectionChunk] = []
    seen_parents: set[str] = set()

    for chunk in all_chunks[start_idx:]:
        current_level = _heading_level(chunk.parent_heading)
        is_new_section = (
            chunk.parent_heading != matched_heading
            and chunk.parent_heading not in seen_parents
            and current_level <= matched_level
        )

        if is_new_section and collected:
            # We've hit the next sibling/parent section — stop
            break

        collected.append(chunk)
        seen_parents.add(chunk.parent_heading)

    # ── Step 4: sort and reassemble ──────────────────────────────────────────
    collected.sort(key=lambda c: (c.start_page, c.chunk_id))

    full_content = "\n\n".join(c.content for c in collected)
    all_pages = sorted({p for c in collected for p in c.pages})
    all_chunk_ids = [c.chunk_id for c in collected]

    return SectionResult(
        query=query,
        heading=matched_heading,
        content=full_content,
        pages=all_pages,
        chunk_ids=all_chunk_ids,
    )


def _heading_level(heading: str) -> int:
    """
    Infer heading depth from its numbering scheme so we know when to stop
    collecting content.

    Examples:
      "1."          → level 1
      "1.1"         → level 2
      "1.1.1"       → level 3
      "Definitions" → level 1  (unnumbered headings treated as top-level)
    """
    match = re.match(r'^(\d+)(\.(\d+))*(\.)?', heading.strip())
    if not match:
        return 1  # unnumbered → treat as top-level

    # Count the dot-separated numeric segments
    numeric_part = re.match(r'^[\d.]+', heading.strip()).group()
    segments = [s for s in numeric_part.split(".") if s.isdigit()]
    return len(segments)

# ── Phase 6: Optional LLM Verification ───────────────────────────────────────

def verify_with_llm(
    result: SectionResult,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> SectionResult:
    """
    Ask a local LLM to confirm the retrieved section matches the query.
    Truncates to 4000 chars to stay within context limits.
    """
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document parser. "
            "Given a section query and a candidate section from a facility agreement, "
            "confirm whether the section matches the query. "
            "If yes, return only the cleaned section text. "
            "If no, respond with exactly: NOT_FOUND"
        )),
        ("human", "Query: {query}\n\nCandidate section:\n{text}"),
    ])

    chain = prompt | llm
    response = chain.invoke({
        "query": result.query,
        "text": result.content[:4000],
    })

    output = response.content.strip()
    confirmed = output != "NOT_FOUND"

    return result.copy(update={
        "verified": confirmed,
        "llm_output": output if confirmed else None,
    })


# ── Loader: build the index once per document ─────────────────────────────────

class FacilityAgreementIndex:
    """
    Build the index once for a PDF, then call .search() as many times
    as needed with different section queries.

    Usage:
        index = FacilityAgreementIndex("agreement.pdf")
        definitions = index.search("Definitions")
        purpose      = index.search("Purpose")
        conditions   = index.search("Conditions Precedent")
    """

    def __init__(
        self,
        pdf_path: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        embed_model: str = "bge-m3",
    ):
        self.pdf_path = pdf_path

        blocks = extract_structured_blocks(pdf_path)
        chunks = chunk_by_section(blocks)
        chunks = split_oversized_chunks(chunks)
        self.vector_store, self.chunks = build_vector_store(
            chunks, base_url=base_url, api_key=api_key, model=embed_model
        )

    def search(
        self,
        section_query: str,
        top_k: int = 3,
        verify: bool = False,
        llm_model: str = "qwen2.5:7b",
    ) -> SectionResult:
        """
        Search for a section by name. Returns the complete section text.

        Args:
            section_query:  Natural language section name, e.g. "Definitions",
                            "Purpose", "Conditions Precedent", "Representations".
            top_k:          Number of candidate chunks to consider.
            verify:         Run an LLM confirmation pass on the result.
            llm_model:      Ollama model name for verification.
        """
        result = search_section(
            query=section_query,
            vector_store=self.vector_store,
            all_chunks=self.chunks,
            top_k=top_k,
        )

        if verify:
            result = verify_with_llm(result, model=llm_model)

        return result


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Build the index once
    index = FacilityAgreementIndex("facility_agreement.pdf")

    # Search for any section by name — works regardless of document language
    for query in ["Definitions", "Purpose", "Conditions Precedent", "Representations"]:
        result = index.search(query)
        print(f"\n{'='*60}")
        print(f"Query   : {result.query}")
        print(f"Section : {result.heading}")
        print(f"Pages   : {result.pages}")
        print(f"Length  : {len(result.content)} chars")
        print(f"\n{result.content[:600]}...")