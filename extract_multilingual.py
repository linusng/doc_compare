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
    vector_store = build_vector_store(chunks)
    candidates = retrieve_definitions_section(vector_store)
    best = pick_best_section(candidates)

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