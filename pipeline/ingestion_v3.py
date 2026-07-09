"""
Smart indexing (v3): TOC-anchored section parsing with real document headings.

Motivation
----------
v2 segments the document page-by-page with an LLM, but the section headings it
produces are only as good as the model's on-the-fly labelling — they don't
reliably match the document's OWN section names. A legal agreement almost always
ships a Table of Contents that lists the canonical section names and their page
numbers. v3 uses that TOC as the source of truth for section structure, so each
chunk's heading is the real section name from the document.

Pipeline
--------
  1. Extract every page's text (reading order).
  2. Detect the TOC page(s) with the LLM (≈1 call).
  3. Parse the TOC into structured entries — {number, title, printed_page} —
     with the LLM (≈1 call). These titles ARE the canonical section names.
  4. ALIGN the TOC's printed page numbers to the PDF parser's page indices.
     The printed numbers rarely equal the parser's 0-based indices (cover pages,
     roman-numeral front matter, inserts). So we do NOT trust the printed number:
     we locate each heading's real text in the body, estimate the constant page
     offset from the anchors we find, and use the printed page only to pick the
     right occurrence when a heading appears more than once.
  5. Slice the body verbatim between consecutive located headings → sections whose
     heading is the actual document section name.
  6. Token-safe split + vector store.

If no usable TOC is found, fall back to a deterministic structural-heading split
(numbered clauses / ARTICLE / SCHEDULE …), which still uses the document's own
heading lines as section names.

Standalone
----------
This module deliberately shares NO processing code with ingestion.py /
ingestion_v2.py — every step is reimplemented here. The only shared import is the
`SectionChunk` data model, so the produced index stays a drop-in for the rest of
the pipeline:

    from pipeline.ingestion_v3 import DocumentIndexV3
    index = DocumentIndexV3.from_pdf("agreement.pdf")
    run_query(index, "What is the repayment schedule?")
"""

import json
import re
import statistics
from dataclasses import dataclass, field

import pymupdf
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import SectionChunk   # shared data model only — no processing reused


# ── Tunables ──────────────────────────────────────────────────────────────────

TARGET_CHUNK_TOKENS = 1000        # retrieval-quality split target
EMBED_TOKEN_CEILING = 8000        # keep well under BGE-M3's 8192 limit
_TOC_MAX_PAGES_TO_CHECK = 20      # only look for a TOC in the first N pages
_MAX_HEADING_LINE_CHARS = 140     # a heading line is short; body prose is not


# ── Token length (tiktoken if available, else a safe estimate) ────────────────

def _make_token_len():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return lambda text: len(enc.encode(text or ""))
    except Exception:   # noqa: BLE001 — offline / tiktoken missing → char estimate
        return lambda text: max(1, len(text or "") // 4)


_token_len = _make_token_len()


# ── Phase 1: page extraction ──────────────────────────────────────────────────

def extract_pages(pdf_path: str) -> list[str]:
    """Return each page's text (reading order); list index = parser page (0-based)."""
    doc = pymupdf.open(pdf_path)
    pages = [page.get_text("text") for page in doc]
    doc.close()
    return pages


# ── Phase 2: TOC page detection (LLM) ─────────────────────────────────────────

def detect_toc_pages(
    pages: list[str],
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    verbose: bool = True,
) -> list[int]:
    """
    Ask the LLM which of the first pages are Table-of-Contents pages. Returns a
    sorted list of parser page indices. Conservative: only pages actually shown to
    the model can be returned, capped at a small consecutive block.
    """
    check = list(range(min(len(pages), _TOC_MAX_PAGES_TO_CHECK)))
    if not check:
        return []

    previews = "\n\n".join(f"--- PAGE {i} ---\n{pages[i][:600]}" for i in check)
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You identify Table of Contents (TOC) pages in a legal document.\n"
            "A TOC page lists section numbers/titles with page-number references "
            "(often with dot leaders), and has NO substantive clause body text. A "
            "cover, title, or signature page is NOT a TOC.\n"
            "Extraction may be messy — titles and page numbers can run together "
            "(e.g. '1.DEFINITIONS2 2.THE FACILITY13'); that is still a TOC.\n"
            "Respond with ONLY a comma-separated list of the PAGE numbers shown "
            "that are TOC pages, or NONE."
        )),
        ("human", "{previews}"),
    ])
    answer = (prompt | llm).invoke({"previews": previews}).content.strip()

    shown = set(check)
    toc = sorted({
        int(tok) for tok in re.findall(r"\d+", answer) if int(tok) in shown
    }) if answer.upper() != "NONE" else []

    # Sanity: a TOC is a small, roughly consecutive block near the front.
    if len(toc) > 6 or (toc and toc[-1] - toc[0] > 6):
        toc = []
    if verbose:
        print(f"      → TOC page(s): {toc or 'none detected'}")
    return toc


# ── Phase 3: TOC parsing (LLM) ────────────────────────────────────────────────

@dataclass
class TocEntry:
    """One line of the table of contents."""
    title: str                       # canonical section name, e.g. "Repayment"
    number: str | None = None        # e.g. "5", "5.1", "ARTICLE V", "SCHEDULE 2"
    printed_page: int | None = None  # the page number printed in the TOC
    level: int = 1

    def label(self) -> str:
        if self.number and self.title:
            return f"{self.number} {self.title}".strip()
        return (self.title or self.number or "").strip()


def parse_toc(
    pages: list[str],
    toc_pages: list[int],
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    verbose: bool = True,
) -> list[TocEntry]:
    """Parse the TOC page text into structured entries with the LLM."""
    if not toc_pages:
        return []
    toc_text = "\n".join(pages[p] for p in toc_pages)

    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You parse a legal document's Table of Contents into JSON. Extract "
            "every entry in order. For each entry return an object with:\n"
            '  "number" : the clause/section number or marker exactly as shown '
            "('5', '5.1', 'ARTICLE V', 'SCHEDULE 2'), or null.\n"
            '  "title"  : the section name text (without the trailing page number '
            "and dot leaders), e.g. 'Repayment'.\n"
            '  "page"   : the page number printed for that entry, or null.\n'
            '  "level"  : 1 for a top-level clause/ARTICLE/SCHEDULE, 2 for x.y, etc.\n'
            "Return ONLY a JSON array, in document order. Do not invent entries."
        )),
        ("human", "{toc}"),
    ])
    raw = (prompt | llm).invoke({"toc": toc_text}).content
    entries = _parse_toc_json(raw)
    if verbose:
        print(f"      → Parsed {len(entries)} TOC entry(ies)"
              + (f"; first: {entries[0].label()!r}" if entries else ""))
    return entries


def _parse_toc_json(raw: str) -> list[TocEntry]:
    """Extract a JSON array of TOC entries from the model's response, tolerantly."""
    m = re.search(r"\[.*\]", raw or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    entries: list[TocEntry] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "") or "").strip()
        number = item.get("number")
        number = str(number).strip() if number not in (None, "") else None
        if not title and not number:
            continue
        page = item.get("page")
        try:
            printed = int(page) if page not in (None, "") else None
        except (TypeError, ValueError):
            printed = None
        try:
            level = int(item.get("level", 1) or 1)
        except (TypeError, ValueError):
            level = 1
        entries.append(TocEntry(title=title, number=number, printed_page=printed, level=level))
    return entries


# ── Phase 4/5: locate headings in the body and align pages ────────────────────

@dataclass
class _DocLine:
    page: int
    text: str


def _build_doc_lines(pages: list[str], exclude: set[int]) -> list[_DocLine]:
    """Flatten non-excluded pages into reading-order lines; drop blank lines and
    short lines that repeat across many pages (running headers/footers)."""
    raw: list[_DocLine] = []
    for pi, text in enumerate(pages):
        if pi in exclude:
            continue
        for ln in text.split("\n"):
            if ln.strip():
                raw.append(_DocLine(page=pi, text=ln.rstrip()))

    # Suppress running headers/footers (short lines repeating on ≥4 pages).
    seen: dict[str, set[int]] = {}
    for dl in raw:
        norm = " ".join(dl.text.split())
        if norm and len(norm) <= 120:
            seen.setdefault(norm, set()).add(dl.page)
    repeating = {t for t, ps in seen.items() if len(ps) >= 4}
    return [dl for dl in raw if " ".join(dl.text.split()) not in repeating]


def _norm(text: str) -> str:
    """Lowercase, strip punctuation to spaces, collapse whitespace."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())


def _heading_matches(line_text: str, entry: TocEntry) -> bool:
    """
    True if `line_text` looks like the body heading line for `entry`.

    Anchored at the start of a short line: the line must begin with the entry's
    number and/or title. Requires real title-word overlap when matching by number,
    so a stray '5' in prose does not match section 5.
    """
    lt = line_text.strip()
    if len(lt) > _MAX_HEADING_LINE_CHARS or sum(c.isalpha() for c in lt) < 2:
        return False
    ln = _norm(lt)
    title_n = _norm(entry.title)
    num_n = _norm(entry.number or "")

    starts_title = bool(title_n) and (ln == title_n or ln.startswith(title_n + " "))
    if starts_title:
        return True

    if num_n and (ln == num_n or ln.startswith(num_n + " ")):
        if not title_n:                      # number-only heading (rare)
            return True
        toks = title_n.split()
        hits = sum(1 for t in toks if t in ln)
        if hits >= max(1, len(toks) // 2):   # number + enough title words
            return True
    return False


@dataclass
class _Located:
    line_index: int          # index into doc_lines
    entry: TocEntry
    heading_text: str        # the real heading line from the document


def align_and_locate(
    entries: list[TocEntry],
    doc_lines: list[_DocLine],
    verbose: bool = True,
) -> tuple[list[_Located], int]:
    """
    Locate each TOC entry's heading in the body and align pages.

    Returns (located, page_offset). `page_offset` = parser_page - printed_page,
    estimated from the entries we could anchor unambiguously; it is used only to
    choose the right occurrence when a heading appears more than once and to keep
    the printed page numbers meaningful for diagnostics.
    """
    # Candidate body-line indices per entry.
    candidates: list[list[int]] = []
    for e in entries:
        candidates.append([i for i, dl in enumerate(doc_lines) if _heading_matches(dl.text, e)])

    # Estimate the constant printed→parser page offset from unambiguous anchors
    # (entries with a printed page and exactly one candidate line).
    samples: list[int] = []
    for e, cands in zip(entries, candidates):
        if e.printed_page is not None and len(cands) == 1:
            samples.append(doc_lines[cands[0]].page - e.printed_page)
    offset = int(round(statistics.median(samples))) if samples else 0

    # Assign in TOC order with a forward cursor so sections stay in document order;
    # break ties by proximity to the expected parser page.
    located: list[_Located] = []
    cursor = 0
    for e, cands in zip(entries, candidates):
        forward = [i for i in cands if i >= cursor]
        pool = forward or cands
        if not pool:
            continue
        if e.printed_page is not None:
            target = e.printed_page + offset
            best = min(pool, key=lambda i: abs(doc_lines[i].page - target))
        else:
            best = pool[0]
        if best < cursor:                    # keep strictly increasing
            continue
        located.append(_Located(best, e, doc_lines[best].text.strip()))
        cursor = best + 1

    if verbose:
        print(f"      → Located {len(located)}/{len(entries)} TOC heading(s) in body; "
              f"printed→parser page offset ≈ {offset}")
    return located, offset


# ── Phase 5: build sections (verbatim slicing) ────────────────────────────────

def _slice(doc_lines: list[_DocLine], start: int, end: int) -> tuple[str, list[int]]:
    span = doc_lines[start:end]
    content = "\n".join(dl.text for dl in span).strip()
    pages = sorted({dl.page for dl in span}) or [0]
    return content, pages


def build_sections(doc_lines: list[_DocLine], located: list[_Located]) -> list[SectionChunk]:
    """Slice the body verbatim between located headings into SectionChunks whose
    heading is the real document section name. Lines before the first heading
    become a 'Preamble and Parties' section. Lossless."""
    sections: list[SectionChunk] = []
    if not doc_lines:
        return sections

    starts = [L.line_index for L in located]
    first = starts[0] if starts else len(doc_lines)
    if first > 0:
        content, pages = _slice(doc_lines, 0, first)
        sections.append(SectionChunk(
            heading="Preamble and Parties", content=content, pages=pages,
            start_page=pages[0], chunk_id=0, heading_level=1,
        ))

    for k, L in enumerate(located):
        end = starts[k + 1] if k + 1 < len(located) else len(doc_lines)
        content, pages = _slice(doc_lines, L.line_index + 1, end)  # body after heading
        _, head_pages = _slice(doc_lines, L.line_index, end)
        heading = L.heading_text.strip() or L.entry.label()
        sections.append(SectionChunk(
            heading=heading, content=content, pages=head_pages,
            start_page=head_pages[0], chunk_id=len(sections),
            heading_level=L.entry.level or 1,
        ))
    return sections


# ── Fallback: deterministic structural-heading split (no TOC) ──────────────────

_STRUCT_RE = re.compile(
    r"^\s*(?:"
    r"article\s+[ivxlcdm]+"
    r"|schedule\s+\w+|annex\s+\w+|appendix\s+\w+|part\s+[ivxlcdm0-9]+"
    r"|(?:section|clause)\s+\d+(?:\.\d+)*"
    r"|§\s*\d+(?:\.\d+)*"
    r"|\d+(?:\.\d+)*\.?"
    r")\b",
    re.IGNORECASE,
)


def _looks_structural(text: str) -> bool:
    t = text.strip()
    return (
        len(t) <= _MAX_HEADING_LINE_CHARS
        and sum(c.isalpha() for c in t) >= 2
        and _STRUCT_RE.match(t) is not None
    )


def heuristic_sections(doc_lines: list[_DocLine]) -> list[SectionChunk]:
    """Split on structural heading lines (numbered clauses / ARTICLE / SCHEDULE),
    using the document's own heading text as the section name. Used when there is
    no usable TOC."""
    located = [
        _Located(i, TocEntry(title=dl.text.strip(), number=None,
                             printed_page=None, level=1), dl.text.strip())
        for i, dl in enumerate(doc_lines) if _looks_structural(dl.text)
    ]
    return build_sections(doc_lines, located)


# ── Phase 6: token-safe split + vector store ──────────────────────────────────

def split_oversized(
    chunks: list[SectionChunk],
    max_tokens: int = TARGET_CHUNK_TOKENS,
    overlap: int = 50,
) -> list[SectionChunk]:
    """Split any chunk whose full_text exceeds max_tokens into token-bounded parts,
    labelled '(part N)', re-numbering chunk_ids sequentially."""
    max_tokens = min(max_tokens, EMBED_TOKEN_CEILING)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_tokens, chunk_overlap=overlap,
        length_function=_token_len, separators=["\n\n", "\n", ". ", " ", ""],
    )
    out: list[SectionChunk] = []
    for c in chunks:
        if _token_len(c.full_text) <= max_tokens:
            out.append(c)
            continue
        for i, sub in enumerate(splitter.split_text(c.content)):
            out.append(SectionChunk(
                heading=f"{c.heading} (part {i + 1})", content=sub, pages=c.pages,
                start_page=c.start_page, chunk_id=0, heading_level=c.heading_level,
            ))
    return [c.model_copy(update={"chunk_id": i}) for i, c in enumerate(out)]


def build_vector_store(
    chunks: list[SectionChunk],
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    model: str = "bge-m3",
) -> InMemoryVectorStore:
    """Embed chunk full_text (heading + body) into an in-memory vector store."""
    embeddings = OpenAIEmbeddings(model=model, base_url=base_url, api_key=api_key)
    docs = [
        Document(
            page_content=c.full_text,
            metadata={
                "chunk_id": c.chunk_id, "heading": c.heading, "pages": c.pages,
                "start_page": c.start_page, "content_length": len(c.content),
                "heading_level": c.heading_level,
            },
        )
        for c in chunks
    ]
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    return store


# ── DocumentIndexV3: standalone, drop-in ──────────────────────────────────────

@dataclass
class DocumentIndexV3:
    """
    TOC-anchored document index. Same duck-typed shape as the other indexes
    (pdf_path, chunks, vector_store), so it is a drop-in for run_extract /
    run_collate / run_query / run_find. Also exposes the parsed TOC and the
    inferred page offset for inspection.
    """
    pdf_path: str
    chunks: list          # list[SectionChunk]
    vector_store: InMemoryVectorStore
    toc_entries: list = field(default_factory=list)
    page_offset: int = 0

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        llm_model: str = "gemma3-27b-it",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        embedding_model: str = "bge-m3",
        target_chunk_tokens: int = TARGET_CHUNK_TOKENS,
        verbose: bool = True,
    ) -> "DocumentIndexV3":
        print(f"[1/5] Extracting page text from: {pdf_path}")
        pages = extract_pages(pdf_path)
        print(f"      → {len(pages)} page(s)")

        print("[2/5] Detecting & parsing Table of Contents (LLM)...")
        toc_pages = detect_toc_pages(pages, llm_model, base_url, api_key, verbose)
        entries = parse_toc(pages, toc_pages, llm_model, base_url, api_key, verbose)

        print("[3/5] Locating section headings in the body & aligning pages...")
        doc_lines = _build_doc_lines(pages, exclude=set(toc_pages))
        sections: list[SectionChunk] = []
        offset = 0
        if entries:
            located, offset = align_and_locate(entries, doc_lines, verbose)
            sections = build_sections(doc_lines, located)

        if len(sections) < 2:
            print("      → No usable TOC sections; falling back to structural-heading split")
            sections = heuristic_sections(doc_lines)
        print(f"      → {len(sections)} section(s); "
              f"headings e.g. {[s.heading for s in sections[:4]]}")

        print("[4/5] Splitting oversized chunks...")
        chunks = split_oversized(sections, max_tokens=target_chunk_tokens)
        print(f"      → {len(chunks)} chunk(s) after token-safe split")

        print("[5/5] Building vector store...")
        vector_store = build_vector_store(chunks, base_url, api_key, embedding_model)
        print(f"      → Vector store ready ({len(chunks)} documents indexed)")

        return cls(pdf_path=pdf_path, chunks=chunks, vector_store=vector_store,
                   toc_entries=entries, page_offset=offset)


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.ingestion_v3 <pdf_path>")
        raise SystemExit(1)

    idx = DocumentIndexV3.from_pdf(sys.argv[1])
    print("\n" + "=" * 60)
    print(f"TOC entries : {len(idx.toc_entries)}  |  page offset ≈ {idx.page_offset}")
    print(f"Sections    : {len(idx.chunks)}")
    for c in idx.chunks[:40]:
        print(f"  [{c.chunk_id:>3}] p{c.start_page:<3} {c.heading[:70]}")
