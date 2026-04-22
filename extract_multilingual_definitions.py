import pymupdf
import statistics
import re
import pydantic
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


# ── Pydantic v1/v2 compat ─────────────────────────────────────────────────────

def pydantic_copy(model, update: dict):
    if int(pydantic.VERSION.split(".")[0]) >= 2:
        return model.model_copy(update=update)
    return model.copy(update=update)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
BGE_M3_MAX_TOKENS = 8192
BGE_M3_CHUNK_OVERLAP = 50


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
    pages: list
    start_page: int
    chunk_id: int = 0
    heading_level: int = 0  # 0 = unknown, 1 = top-level (e.g. "1."), 2 = sub (e.g. "1.1")

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class ExtractionResult(BaseModel):
    heading: str
    content: str
    pages: list
    chunk_id: int
    verified: bool = False
    llm_output: str = None


# ── Phase 1: Structured Extraction ───────────────────────────────────────────

def extract_structured_blocks(pdf_path: str) -> list:
    """Extract text blocks from PDF, preserving font metadata for heading detection."""
    doc = pymupdf.open(pdf_path)
    blocks = []

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

def _get_heading_level(text: str) -> int:
    """
    Determine heading level from section numbering.
    '1.' or '1'       → level 1
    '1.1' or '1.1.'   → level 2
    '1.1.1'           → level 3
    No number          → 0
    """
    m = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s', text.strip())
    if not m:
        return 0
    parts = m.group(1).split(".")
    parts = [p for p in parts if p]  # drop empty from trailing dot
    return len(parts)


def detect_headings(blocks: list) -> list:
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
        updated.append(pydantic_copy(block, {"is_heading": is_heading}))

    return updated


def chunk_by_section(blocks: list) -> list:
    """
    Produce one SectionChunk per section: heading + all body text beneath it.
    Preserves page metadata and heading level throughout.
    """
    blocks = detect_headings(blocks)
    chunks = []
    current_heading = ""
    current_body = []
    current_pages = set()

    def flush(chunk_id: int):
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


# ── Phase 3: Token-safe Splitting ─────────────────────────────────────────────

def token_len(text: str) -> int:
    return len(_tokenizer.encode(text, add_special_tokens=True))


def split_oversized_chunks(
    chunks: list,
    max_tokens: int = BGE_M3_MAX_TOKENS,
    chunk_overlap: int = BGE_M3_CHUNK_OVERLAP,
) -> list:
    """
    Any SectionChunk whose full_text exceeds max_tokens is split into
    smaller sub-chunks using RecursiveCharacterTextSplitter with the
    real BGE-M3 tokenizer as the length function.
    Heading and page metadata are propagated to every sub-chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens,
        chunk_overlap=chunk_overlap,
        length_function=token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    result = []

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
                heading_level=chunk.heading_level,
            ))

    # Re-assign chunk_ids after potential splits
    for idx, chunk in enumerate(result):
        result[idx] = pydantic_copy(chunk, {"chunk_id": idx})

    return result


# ── Phase 4: Embed into InMemoryVectorStore ───────────────────────────────────

def build_vector_store(
    chunks: list,
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
                "heading_level": chunk.heading_level,
            },
        )
        for chunk in chunks
    ]

    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(docs)
    return vector_store


# ── Phase 5: Multilingual Semantic Retrieval ──────────────────────────────────

DEFINITIONS_QUERIES = [
    "Definitions and Interpretation",
    "Definition of terms used in this agreement",
    "Interpretation clause",
    "Meanings of defined terms",
]


def retrieve_definitions_section(
    vector_store: InMemoryVectorStore,
    top_k: int = 5,
) -> list:
    """
    Run multiple query angles and merge results for robustness.
    BGE-M3's cross-lingual alignment handles French, German, Chinese, etc.
    Deduplicates by chunk_id and sorts by page order.
    """
    seen = {}

    for query in DEFINITIONS_QUERIES:
        results = vector_store.similarity_search(query, k=top_k)
        for doc in results:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen:
                seen[chunk_id] = doc

    return sorted(seen.values(), key=lambda d: d.metadata["start_page"])


def pick_best_section(candidates: list) -> Document:
    """
    Score candidates and return the most likely Definitions section.
    Prefers section-numbered headings with substantial content.
    """
    scored = []

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
    return scored[0][1]


# ── Phase 5b: Gather Full Section ─────────────────────────────────────────────

def _base_heading(heading: str) -> str:
    """
    Strip '(part N)' suffixes added by split_oversized_chunks,
    so we can match split fragments back to their parent.
    """
    return re.sub(r'\s*\(part \d+\)\s*$', '', heading).strip()


def _extract_section_number(heading: str) -> str | None:
    """
    Return the leading section number from a heading, e.g. '1' from '1. Definitions'.
    Returns None if no section number is found.
    """
    heading = _base_heading(heading)
    m = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s', heading.strip())
    if m:
        return m.group(1)
    return None


def _is_child_section(parent_num: str, candidate_heading: str) -> bool:
    """
    Check if a candidate heading is a sub-section of parent_num.
    E.g. parent_num='1' → '1.1', '1.2', '1.15' are children; '2', '2.1' are not.
    Also matches split parts of the same heading (via _base_heading).
    """
    cand_num = _extract_section_number(candidate_heading)
    if cand_num is None:
        return False
    # Same section number (e.g. both are "1" — split parts)
    if cand_num == parent_num:
        return True
    # True child: starts with parent_num + "."
    return cand_num.startswith(parent_num + ".")


def gather_full_section(
    best_doc: Document,
    all_chunks: list[SectionChunk],
) -> list[SectionChunk]:
    """
    Given the best-matched chunk, collect ALL chunks that belong to the
    same logical section. This handles:
      - Split parts:    '1. Definitions (part 1)', '... (part 2)', etc.
      - Sub-sections:   '1.1 Defined Terms', '1.2 Interpretation', etc.
      - Body chunks between headings that have no section number.

    Strategy:
      1. Find the best chunk's position in all_chunks by chunk_id.
      2. Determine its top-level section number (e.g. '1').
      3. Walk forward from that position, collecting chunks as long as they
         are children/parts of the same section OR have no section number
         (i.e. body text that was promoted to a "heading" by bold detection).
      4. Stop when we hit the next top-level section that isn't a child.
    """
    best_id = best_doc.metadata["chunk_id"]
    best_heading = best_doc.metadata["heading"]
    parent_num = _extract_section_number(best_heading)

    # Find the start index in all_chunks
    start_idx = None
    for i, chunk in enumerate(all_chunks):
        if chunk.chunk_id == best_id:
            start_idx = i
            break

    if start_idx is None:
        # Fallback: just return the single chunk
        return [c for c in all_chunks if c.chunk_id == best_id]

    # If we couldn't parse a section number, fall back to collecting
    # only chunks with the same base heading (split parts).
    if parent_num is None:
        base = _base_heading(best_heading)
        return [
            c for c in all_chunks
            if _base_heading(c.heading) == base
        ]

    # Walk forward from start_idx and collect children / same-section chunks
    collected = []
    for i in range(start_idx, len(all_chunks)):
        chunk = all_chunks[i]
        heading = chunk.heading

        if i == start_idx:
            collected.append(chunk)
            continue

        # Check if this chunk belongs to the same section
        cand_num = _extract_section_number(heading)

        if cand_num is not None:
            if _is_child_section(parent_num, heading):
                collected.append(chunk)
            else:
                # Hit a sibling or higher-level section — stop
                break
        else:
            # No section number in heading — this is likely bold body text
            # or a non-numbered sub-heading within our section. Include it
            # only if it appears before the next numbered section.
            collected.append(chunk)

    # Also walk backward to catch any earlier split parts (e.g. if the
    # best match was part 2 but part 1 exists before it)
    base = _base_heading(best_heading)
    for i in range(start_idx - 1, -1, -1):
        chunk = all_chunks[i]
        cand_num = _extract_section_number(chunk.heading)

        if _base_heading(chunk.heading) == base:
            # Same heading, earlier split part
            collected.insert(0, chunk)
        elif cand_num and _is_child_section(parent_num, chunk.heading):
            collected.insert(0, chunk)
        else:
            break

    return collected


def merge_section_chunks(chunks: list[SectionChunk]) -> ExtractionResult:
    """
    Merge a list of SectionChunks into a single ExtractionResult with
    combined content and unified page list.
    """
    if not chunks:
        raise ValueError("No chunks to merge")

    all_pages = set()
    parts = []

    for chunk in chunks:
        all_pages.update(chunk.pages)
        # Include the heading for sub-sections to preserve structure,
        # but skip duplicate headings for split parts
        if chunk.heading and "(part " not in chunk.heading:
            parts.append(f"\n{chunk.heading}\n{chunk.content}")
        else:
            parts.append(chunk.content)

    combined_content = "\n".join(parts).strip()

    return ExtractionResult(
        heading=_base_heading(chunks[0].heading),
        content=combined_content,
        pages=sorted(all_pages),
        chunk_id=chunks[0].chunk_id,
    )


# ── Phase 6: LLM Verification ─────────────────────────────────────────────────

def verify_with_llm(
    doc: Document,
    model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> tuple:
    """
    Ask a local LLM to confirm the retrieved chunk is the Definitions section.
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
    llm_model: str = "gemma3-27b-it",
    ollama_base_url: str = "http://localhost:11434/v1",
    ollama_api_key: str = "ollama",
    embedding_model: str = "bge-m3",
) -> ExtractionResult:
    """
    Full pipeline:
      PDF → structured blocks → section chunks → token-safe splits
          → BGE-M3 embeddings → semantic retrieval
          → gather full section → merge → ExtractionResult
    """
    print(f"[1/6] Extracting blocks from: {pdf_path}")
    blocks = extract_structured_blocks(pdf_path)
    print(f"      → {len(blocks)} blocks extracted")

    print("[2/6] Chunking by section...")
    chunks = chunk_by_section(blocks)
    print(f"      → {len(chunks)} sections found")

    print("[3/6] Splitting oversized chunks...")
    chunks = split_oversized_chunks(chunks)
    print(f"      → {len(chunks)} chunks after token-safe split")

    print("[4/6] Building vector store...")
    vector_store = build_vector_store(
        chunks,
        base_url=ollama_base_url,
        api_key=ollama_api_key,
        model=embedding_model,
    )

    print("[5/6] Retrieving definitions section...")
    candidates = retrieve_definitions_section(vector_store)
    best = pick_best_section(candidates)

    print("[6/6] Gathering full section...")
    section_chunks = gather_full_section(best, chunks)
    print(f"      → {len(section_chunks)} chunks in full section")
    result = merge_section_chunks(section_chunks)

    if verify:
        print("      Running LLM verification...")
        # For verification, wrap the merged result back into a Document
        verify_doc = Document(
            page_content=result.content,
            metadata={"heading": result.heading, "pages": result.pages},
        )
        confirmed, llm_output = verify_with_llm(
            verify_doc,
            model=llm_model,
            base_url=ollama_base_url,
            api_key=ollama_api_key,
        )
        result = pydantic_copy(result, {
            "verified": confirmed,
            "llm_output": llm_output,
        })

    return result


# ── Usage ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = extract_definitions_section(
        pdf_path="facility_agreement.pdf",
        verify=True,
    )

    print("\n── Extraction Result ──────────────────────────────────────")
    print(f"Section  : {result.heading}")
    print(f"Pages    : {result.pages}")
    print(f"Chunks   : gathered into single result")
    print(f"Verified : {result.verified}")
    print(f"\n{result.content[:800]}")