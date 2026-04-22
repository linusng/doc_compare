import pymupdf
import statistics
import re
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate


# ── Models ────────────────────────────────────────────────────────────────────

class TextBlock(BaseModel):
    text: str
    page: int
    bbox: tuple[float, float, float, float]
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

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class ExtractionResult(BaseModel):
    heading: str
    content: str
    pages: list[int]
    chunk_id: int
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
        updated.append(block.model_copy(update={"is_heading": is_heading}))

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


# ── Phase 3: Embed into InMemoryVectorStore ───────────────────────────────────

def build_vector_store(
    chunks: list[SectionChunk],
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model: str = "bge-m3",
) -> InMemoryVectorStore:
    """
    Embed all section chunks using BGE-M3 and load into LangChain's
    InMemoryVectorStore. No external DB required.
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
                "pages": chunk.pages,
                "start_page": chunk.start_page,
                "content_length": len(chunk.content),
            },
        )
        for chunk in chunks
    ]

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(docs)
    return vector_store


# ── Phase 4: Multilingual Semantic Retrieval ─────────────────────────────────

DEFINITIONS_QUERIES = [
    "Definitions and Interpretation",
    "Definition of terms used in this agreement",
    "Interpretation clause",
    "Meanings of defined terms",
]


def retrieve_definitions_section(
    vector_store: InMemoryVectorStore,
    top_k: int = 5,
) -> list[Document]:
    """
    Run multiple query angles and merge results for robustness.
    BGE-M3's cross-lingual alignment handles French, German, Chinese, etc.
    Deduplicates by chunk_id and sorts by page order.
    """
    seen: dict[int, Document] = {}

    for query in DEFINITIONS_QUERIES:
        results = vector_store.similarity_search(query, k=top_k)
        for doc in results:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen:
                seen[chunk_id] = doc

    return sorted(seen.values(), key=lambda d: d.metadata["start_page"])


def pick_best_section(candidates: list[Document]) -> Document:
    """
    Score candidates and return the most likely Definitions section.
    Prefers section-numbered headings with substantial content.
    """
    scored: list[tuple[float, Document]] = []

    for doc in candidates:
        heading = doc.metadata.get("heading", "")
        content_length = doc.metadata.get("content_length", len(doc.page_content))
        score = 0.0

        if re.search(r'^\d+[\.\d]*', heading.strip()):
            score += 2.0                            # numbered section heading
        score += min(content_length / 500, 3.0)     # reward longer content
        if content_length < 100:
            score -= 2.0                            # penalise stubs

        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


# ── Phase 5: Optional LLM Verification ───────────────────────────────────────

def verify_with_llm(
    doc: Document,
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> tuple[bool, str]:
    """
    Ask a local LLM to confirm the retrieved chunk is the Definitions section.
    Returns (is_confirmed, cleaned_text).
    """
    llm = ChatOpenAI(model=model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document parser. "
            "Given a candidate text block from a facility agreement, "
            "confirm if it is the Definitions and Interpretation section. "
            "If yes, return only the cleaned section text. "
            "If no, respond with exactly: NOT_FOUND"
        )),
        ("human", "{text}"),
    ])

    chain = prompt | llm
    result = chain.invoke({"text": doc.page_content[:4000]})
    output = result.content.strip()
    return output != "NOT_FOUND", output

from langchain_text_splitters import RecursiveCharacterTextSplitter

# BGE-M3 max = 8192 tokens. At ~4 chars/token, 8192 * 4 = 32768 chars.
# Use 0.85 safety margin → 27,500 chars max per chunk.
BGE_M3_MAX_CHARS = 27_500
BGE_M3_CHUNK_OVERLAP = 500


def split_oversized_chunks(
    chunks: list[SectionChunk],
    max_chars: int = BGE_M3_MAX_CHARS,
    chunk_overlap: int = BGE_M3_CHUNK_OVERLAP,
) -> list[SectionChunk]:
    """
    Any SectionChunk whose full_text exceeds max_chars is split into
    smaller sub-chunks using RecursiveCharacterTextSplitter.
    Heading and page metadata are propagated to every sub-chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result: list[SectionChunk] = []

    for chunk in chunks:
        if len(chunk.full_text) <= max_chars:
            result.append(chunk)
            continue

        sub_texts = splitter.split_text(chunk.content)

        for i, sub_text in enumerate(sub_texts):
            result.append(SectionChunk(
                heading=f"{chunk.heading} (part {i + 1})",
                content=sub_text,
                pages=chunk.pages,
                start_page=chunk.start_page,
                chunk_id=0,          # re-assigned in build_vector_store
            ))

    # Re-assign chunk_ids after potential splits
    for idx, chunk in enumerate(result):
        result[idx] = chunk.copy(update={"chunk_id": idx})

    return result




def retrieve_definitions_section(
    vector_store: InMemoryVectorStore,
    top_k: int = 10,                        # increase to cast a wider net
) -> list[Document]:
    seen: dict[int, Document] = {}

    for query in DEFINITIONS_QUERIES:
        results = vector_store.similarity_search(query, k=top_k)
        for doc in results:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen:
                seen[chunk_id] = doc

    return sorted(seen.values(), key=lambda d: d.metadata["start_page"])


def pick_best_section(
    candidates: list[Document],
    all_chunks: list[SectionChunk],
) -> ExtractionResult:
    """
    1. Find the single best anchor chunk (the heading match).
    2. Walk forward through all_chunks from that anchor, collecting
       consecutive chunks that belong to the same section — i.e. sub-sections
       like 1.1, 1.2 … or continuation parts — until a new top-level
       section heading is encountered.
    3. Stitch all collected chunks into one ExtractionResult.
    """

    # ── Step 1: score candidates to find the anchor ───────────────────────
    scored: list[tuple[float, Document]] = []

    for doc in candidates:
        heading = doc.metadata.get("heading", "")
        content_length = doc.metadata.get("content_length", len(doc.page_content))
        score = 0.0

        if re.search(r'^\d+[\.\d]*', heading.strip()):
            score += 2.0
        score += min(content_length / 500, 3.0)
        if content_length < 100:
            score -= 2.0

        scored.append((score, doc))

    scored.sort(key=lambda x: -x[0])
    anchor = scored[0][1]
    anchor_id = anchor.metadata["chunk_id"]

    # ── Step 2: detect the top-level section number of the anchor ─────────
    anchor_heading = anchor.metadata.get("heading", "")
    # e.g. "1.1 Definitions..." → top-level section = "1"
    top_level_match = re.match(r'^(\d+)[\.\s]', anchor_heading.strip())
    top_level_prefix = top_level_match.group(1) + "." if top_level_match else None

    # ── Step 3: walk forward from anchor, collecting sibling chunks ───────
    collected: list[SectionChunk] = []
    collecting = False

    for chunk in all_chunks:
        if chunk.chunk_id == anchor_id:
            collecting = True

        if not collecting:
            continue

        heading = chunk.heading.strip()

        if chunk.chunk_id != anchor_id:
            # Stop when we hit a NEW top-level section
            # e.g. stop at "2." or "2.1" but continue through "1.1", "1.2"
            new_top = re.match(r'^(\d+)[\.\s]', heading)
            if new_top:
                new_prefix = new_top.group(1) + "."
                if top_level_prefix and new_prefix != top_level_prefix:
                    break
            # Also stop on non-numbered bold headings that signal a new section
            elif heading and not heading.startswith("("):
                # Heuristic: short bold-style line with no parent prefix = new section
                if len(heading) < 80 and chunk.chunk_id > anchor_id + 1:
                    break

        collected.append(chunk)

    # ── Step 4: stitch collected chunks into one result ───────────────────
    all_pages = sorted({p for chunk in collected for p in chunk.pages})
    full_content = "\n\n".join(chunk.full_text for chunk in collected)

    return ExtractionResult(
        heading=anchor_heading,
        content=full_content,
        pages=all_pages,
        chunk_id=anchor_id,
    )








# ── Main Pipeline ─────────────────────────────────────────────────────────────

def extract_definitions_section(
    pdf_path: str,
    verify: bool = False,
    llm_model: str = "qwen2.5:7b",
) -> ExtractionResult:
    """
    Full pipeline: PDF → structured blocks → section chunks →
    BGE-M3 embeddings → semantic retrieval → ExtractionResult.
    """
    blocks = extract_structured_blocks(pdf_path)
    chunks = chunk_by_section(blocks)
    chunks = split_oversized_chunks(chunks)   # ← add this line
    vector_store = build_vector_store(chunks)
    candidates = retrieve_definitions_section(vector_store)
    # best = pick_best_section(candidates)
    result = pick_best_section(candidates, all_chunks=chunks)  # ← pass chunks

    result = ExtractionResult(
        heading=best.metadata["heading"],
        content=best.page_content,
        pages=best.metadata["pages"],
        chunk_id=best.metadata["chunk_id"],
    )

    if verify:
        confirmed, llm_output = verify_with_llm(best, model=llm_model)
        result = result.model_copy(update={
            "verified": confirmed,
            "llm_output": llm_output,
        })

    return result


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = extract_definitions_section(
        "facility_agreement.pdf",
        verify=True,
    )

    print(f"Section : {result.heading}")
    print(f"Pages   : {result.pages}")
    print(f"Verified: {result.verified}")
    print(f"\n{result.content[:800]}")