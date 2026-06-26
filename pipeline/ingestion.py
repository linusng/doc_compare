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
from .section_utils import _extract_named_prefix, _extract_section_number


# ── Tokeniser (BGE-M3) ────────────────────────────────────────────────────────

_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
BGE_M3_MAX_TOKENS = 8192          # hard embedding-capacity ceiling
BGE_M3_CHUNK_OVERLAP = 50

# Retrieval-oriented target chunk size.  This is deliberately far below the
# 8192-token embedding ceiling: embedding a very large section produces a
# diluted, unfocused vector, so big sections (US "Defined Terms", long covenant
# clauses, etc.) retrieve poorly as one blob.  Splitting them into ~this many
# tokens keeps each vector focused and dramatically improves retrieval.
# Tune via DocumentIndex.from_pdf(target_chunk_tokens=...).
TARGET_CHUNK_TOKENS = 1000


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
        page_height = float(page.rect.height)

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
                    page_height=page_height,
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


# ── Step 2b: Repeating block filter (headers / footers) ───────────────────────

def filter_repeating_blocks(
    blocks: list[TextBlock],
    min_pages: int = 4,
    max_text_len: int = 120,
) -> list[TextBlock]:
    """
    Remove blocks whose text repeats across many pages — these are page
    headers, footers, running titles, and page numbers that PyMuPDF returns
    as ordinary blocks on every page.

    A block is considered a repeating artefact when:
    - Its normalised text is identical across ≥ min_pages pages, AND
    - Its text is short (≤ max_text_len chars) — real content doesn't repeat
      verbatim across many pages.

    Parameters
    ----------
    min_pages    : Minimum number of pages a text must appear on to be
                   considered a repeating artefact (default 4).  Set higher
                   for shorter documents where 4 pages is a lot.
    max_text_len : Only consider short blocks as candidates.  Long blocks
                   (tables, definitions) are never artefacts (default 120).
    """
    from collections import Counter

    # Count how many distinct pages each normalised text appears on
    text_page_pairs: set[tuple[str, int]] = set()
    for b in blocks:
        normalised = " ".join(b.text.split())   # collapse whitespace
        if normalised and len(normalised) <= max_text_len:
            text_page_pairs.add((normalised, b.page))

    # text → number of distinct pages it appears on
    page_counts: Counter[str] = Counter()
    for text, _ in text_page_pairs:
        page_counts[text] += 1

    repeating: set[str] = {
        text for text, count in page_counts.items() if count >= min_pages
    }

    if repeating:
        samples = sorted(repeating)[:5]
        print(f"      → Suppressed {len(repeating)} repeating header/footer block(s): {samples}")

    return [
        b for b in blocks
        if " ".join(b.text.split()) not in repeating
    ]


# ── Step 2c: Positional header / footer filter ────────────────────────────────

def filter_positional_headers_footers(
    blocks: list[TextBlock],
    min_pages: int = 4,
    top_frac: float = 0.08,
    bottom_frac: float = 0.92,
    max_text_len: int = 120,
    band_buckets: int = 100,
) -> list[TextBlock]:
    """
    Remove running headers/footers detected by POSITION rather than by text.

    The text-based `filter_repeating_blocks` only catches blocks whose text is
    identical across pages.  It misses running headers/footers whose text VARIES
    per page — page numbers, dates, and (critically) running titles that name
    the current clause (e.g. a bold "2 THE FACILITY" header appearing on a page
    that still contains Definitions text).  Those varying blocks survive, get
    promoted to headings, and split a section in two — which is why a long
    section like Definitions can be cut off halfway.

    A block is treated as a running header/footer when ALL of:
      - it is short (≤ max_text_len chars) — body paragraphs are never headers,
      - it sits in the top margin band (its bottom edge is above top_frac of the
        page) OR the bottom margin band (its top edge is below bottom_frac), AND
      - a block occupies that same margin + vertical position on ≥ min_pages
        distinct pages.  Recurrence at a fixed position across pages is the
        signature of a running element; a genuine one-off section heading does
        not recur at the same band position on many pages.

    Parameters
    ----------
    top_frac / bottom_frac : margin band thresholds as a fraction of page
                             height.  Kept tight (top 8% / bottom 8%) so that
                             real section headings, which sit in the body area,
                             are not caught.
    band_buckets           : vertical resolution for position matching (100 →
                             buckets of 1% of page height).
    """
    from collections import Counter

    def position_key(b: TextBlock) -> tuple[str, int] | None:
        h = b.page_height
        if h <= 0:
            return None
        y0, y1 = b.bbox[1], b.bbox[3]
        center_frac = ((y0 + y1) / 2.0) / h
        bucket = round(center_frac * band_buckets)
        if y1 <= h * top_frac:
            return ("top", bucket)
        if y0 >= h * bottom_frac:
            return ("bottom", bucket)
        return None

    # Count distinct pages per (side, vertical-bucket) for short margin blocks.
    key_pages: dict[tuple[str, int], set[int]] = {}
    for b in blocks:
        if len(" ".join(b.text.split())) > max_text_len:
            continue
        key = position_key(b)
        if key is not None:
            key_pages.setdefault(key, set()).add(b.page)

    running_keys = {k for k, pages in key_pages.items() if len(pages) >= min_pages}

    if not running_keys:
        print("      → No positional header/footer bands detected")
        return blocks

    kept: list[TextBlock] = []
    removed_samples: list[str] = []
    for b in blocks:
        if len(" ".join(b.text.split())) <= max_text_len and position_key(b) in running_keys:
            if len(removed_samples) < 5:
                removed_samples.append(repr(b.text[:40]))
            continue
        kept.append(b)

    print(f"      → Suppressed positional header/footer blocks "
          f"({len(blocks) - len(kept)} block(s)); samples: {removed_samples}")
    return kept


# ── Step 3: Heading detection and section chunking ────────────────────────────

def _get_heading_level(text: str) -> int:
    """
    Infer heading depth from the section number prefix, using the shared
    US/UK-aware number parser so e.g. 'Section 1.01' and 'ARTICLE I' are
    levelled consistently.
    '1.'/'ARTICLE I' → 1,  '1.1'/'Section 1.01' → 2,  '1.1.1' → 3,  none → 0.
    """
    num = _extract_section_number(text)
    if not num:
        return 0
    return len([p for p in num.split(".") if p])


# Minimum characters a block must have to qualify as a heading.
# Blocks like "4", "•", or "Move To" are far too short to be real section
# headings — they are artefacts from page numbers, bullet markers, or
# navigation labels that survive the repeating-block filter because they
# appear on fewer than min_pages pages.
_MIN_HEADING_CHARS = 4

# A heading must contain at least this many letters (not just digits/symbols).
# This catches pure page numbers ("42"), roman numerals ("IV"), and lone
# punctuation marks that bold formatting makes look like headings.
_MIN_HEADING_LETTERS = 2


# Structural heading markers (US + UK).  Many US credit agreements do NOT bold
# their Section headings (so font-based detection misses them) and often run the
# heading inline with the body ("Section 2.01 Commitment.  Subject to …").
# Detecting these by pattern — in addition to font/bold — is what lets the
# document chunk into real per-Section pieces instead of giant ARTICLE blobs.
_STRUCT_HEADING_RE = re.compile(
    r'^\s*(?:'
    r'article\s+[ivxlcdm]+'              # ARTICLE I, ARTICLE IV
    r'|(?:section|sec\.)\s+\d+(?:\.\d+)+'  # Section 1.01, Sec. 2.03, 1.0.1
    r'|§\s*\d+(?:\.\d+)*'                  # § 2.3
    r'|\d+(?:\.\d+)+'                      # 1.01 / 1.0.1 at the line start
    r')\b',
    re.IGNORECASE,
)

# A pattern-detected heading must be at least this short to be treated as a
# standalone heading line; longer blocks are run-in headings that should have
# been split by split_structural_headings first.
_MAX_PATTERN_HEADING_CHARS = 160

# Run-in splitting: a block longer than this that begins with a structural
# marker is treated as "heading. body" and split at the title's end.
_RUNIN_MIN_BLOCK_CHARS = 120
_RUNIN_MAX_HEADING_CHARS = 140


def split_structural_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Split run-in section headings into a separate heading block + body block.

    Many US agreements write the section heading inline with its first
    paragraph, e.g.:

        "Section 2.01  Commitment.  Subject to the terms hereof, each Lender
         severally agrees to make Loans …"

    PyMuPDF returns that as one long block.  Because it is long, font/pattern
    heading detection won't (and shouldn't) treat the whole thing as a heading.
    This step cuts it into:

        heading: "Section 2.01  Commitment."
        body:    "Subject to the terms hereof, each Lender …"

    so the heading can be detected and the section chunked correctly.  Blocks
    that don't start with a structural marker, or that are already short, are
    returned unchanged.
    """
    out: list[TextBlock] = []
    n_split = 0

    for b in blocks:
        stripped = b.text.strip()
        m = _STRUCT_HEADING_RE.match(stripped)

        if not m or len(stripped) <= _RUNIN_MIN_BLOCK_CHARS:
            out.append(b)
            continue

        # Long block that begins with a marker → run-in heading.  The title runs
        # from the end of the marker to the first sentence terminator.
        rest = stripped[m.end():]
        pm = re.search(r'\.\s', rest)
        if not pm:
            out.append(b)
            continue

        cut = m.end() + pm.end()
        head = stripped[:cut].strip()
        body = stripped[cut:].strip()
        if not body or len(head) > _RUNIN_MAX_HEADING_CHARS:
            out.append(b)
            continue

        out.append(pydantic_copy(b, {"text": head}))
        out.append(pydantic_copy(b, {"text": body}))
        n_split += 1

    if n_split:
        print(f"      → Split {n_split} run-in section heading(s)")
    return out


def _is_structural_heading(text: str) -> bool:
    """True if `text` begins with a recognised section marker and is short
    enough to be a standalone heading line (not a run-in paragraph)."""
    stripped = text.strip()
    return (
        len(stripped) <= _MAX_PATTERN_HEADING_CHARS
        and _STRUCT_HEADING_RE.match(stripped) is not None
    )


# Enumerated list markers — '(1)', '(2)', '(a)', '(i)', etc. — that open a
# parenthesis. In legal documents these introduce list ITEMS (notably the
# numbered parties in a preamble: "(1) <corporate name>  (2) <bank name>"),
# not section headings. Promoting them to headings splits the preamble/parties
# block into fragments, so they are explicitly excluded from heading detection.
_LIST_ITEM_RE = re.compile(r'^\s*\(\s*(?:\d+|[a-z]|[ivxlcdm]+)\s*\)', re.IGNORECASE)


def _is_list_item(text: str) -> bool:
    """True if the block opens with a parenthesised enumerator like '(1)' / '(a)'."""
    return _LIST_ITEM_RE.match(text) is not None


def _is_valid_heading_text(text: str) -> bool:
    """
    Return True only if `text` looks like a genuine section heading.

    Rejects:
    - Strings shorter than _MIN_HEADING_CHARS characters
    - Strings with fewer than _MIN_HEADING_LETTERS alphabetic characters
      (pure numbers, page numbers, symbols)
    - Strings that are purely numeric after stripping whitespace/punctuation
    - Parenthesised list items ('(1) …', '(a) …') — list entries, not headings
    """
    stripped = text.strip()
    if len(stripped) < _MIN_HEADING_CHARS:
        return False
    letter_count = sum(1 for c in stripped if c.isalpha())
    if letter_count < _MIN_HEADING_LETTERS:
        return False
    if _is_list_item(stripped):
        return False
    return True


# Margin bands (as a fraction of page height) in which a block is considered to
# be in the header/footer area rather than the body. A genuine section heading
# lives in the body; page numbers, running titles and footer notes like
# "<page no.> Internal" sit in these margins. Blocks centred inside a margin band
# are never promoted to headings, which stops a per-page footer such as
# "2 Internal" from being mis-parsed as section "2" and splitting a section.
_HEADING_TOP_MARGIN_FRAC = 0.08      # top 8% of the page
_HEADING_BOTTOM_MARGIN_FRAC = 0.90   # bottom 10% of the page


def _in_page_margin(
    block: TextBlock,
    top_frac: float = _HEADING_TOP_MARGIN_FRAC,
    bottom_frac: float = _HEADING_BOTTOM_MARGIN_FRAC,
) -> bool:
    """
    True if the block's vertical centre sits in the top or bottom margin band of
    its page — i.e. the header/footer zone where headings never appear.

    Returns False when page height is unknown (0), so detection degrades to the
    prior font/structure behaviour rather than silently dropping headings.
    """
    h = block.page_height
    if h <= 0:
        return False
    center_frac = ((block.bbox[1] + block.bbox[3]) / 2.0) / h
    return center_frac <= top_frac or center_frac >= bottom_frac


def detect_headings(blocks: list[TextBlock]) -> list[TextBlock]:
    """
    Mark blocks as headings based on font size and boldness, with a validity
    guard that prevents artefacts from being promoted to headings.

    A block is a heading if it passes _is_valid_heading_text (minimum length
    and letter count), is NOT in a page margin band (see _in_page_margin), AND
    any of:
    - Its font size exceeds 1.1× the document median, OR
    - It is bold and short (< 120 characters), OR
    - It begins with a structural section marker (ARTICLE I, Section 1.01,
      § 2.3, 1.0.1) and is short enough to be a standalone heading line.
      This last signal is essential for US agreements that do not bold their
      Section headings.

    The margin guard is what stops a recurring footer like "<page no.> Internal"
    — whose per-page text varies, so the text-based filters miss it — from being
    promoted to a heading and parsed as a spurious section number.
    """
    font_sizes = [b.font_size for b in blocks if b.font_size > 0]
    if not font_sizes:
        return blocks

    median_size = statistics.median(font_sizes)
    updated = []
    n_margin_suppressed = 0
    for block in blocks:
        visually_a_heading = (
            block.font_size > median_size * 1.1
            or (block.is_bold and len(block.text) < 120)
        )
        structurally_a_heading = _is_structural_heading(block.text)
        in_margin = _in_page_margin(block)
        is_heading = (
            (visually_a_heading or structurally_a_heading)
            and _is_valid_heading_text(block.text)
            and not in_margin
        )
        if in_margin and (visually_a_heading or structurally_a_heading) \
                and _is_valid_heading_text(block.text):
            n_margin_suppressed += 1
        updated.append(pydantic_copy(block, {"is_heading": is_heading}))

    if n_margin_suppressed:
        print(f"      → [diag] Suppressed {n_margin_suppressed} margin block(s) "
              f"that would otherwise have been promoted to headings")
    return updated


def _is_structural_section_heading(text: str) -> bool:
    """
    True if the heading is a real structural section anchor — a numbered heading
    ('1.', 'ARTICLE I', 'Section 1.01', '§ 2.3') or a named section ('Schedule
    2', 'Annex A'). These are the headings allowed to open a new section while
    still in the preamble; visual-only (bold/large) headings are not.
    """
    return (
        _extract_section_number(text) is not None
        or _extract_named_prefix(text) is not None
    )


# The operative opener of an agreement — "THIS AGREEMENT is dated …",
# "THIS DEED OF GUARANTEE …", "THIS FACILITY AGREEMENT is made …". Within the
# preamble region this line, not the cover title, should anchor the preamble
# chunk's heading so the chunk leads with (and is retrievable by) the opener.
_PREAMBLE_OPENER_RE = re.compile(
    r"^\s*THIS\s+[\w\s\-,']*\b(?:AGREEMENT|DEED)\b",
    re.IGNORECASE,
)


def _is_preamble_opener(text: str) -> bool:
    return _PREAMBLE_OPENER_RE.match(text) is not None


def chunk_by_section(blocks: list[TextBlock]) -> list[SectionChunk]:
    """
    Produce one SectionChunk per section: heading + all body text beneath it.
    Page metadata is accumulated across all blocks in the section.

    Preamble handling: everything before the FIRST structural heading (the
    recitals and numbered parties — "THIS AGREEMENT … BETWEEN:- (1) … (2) …")
    is kept as a SINGLE preamble chunk. While still in this region, a
    visual-only heading (bold/large text such as "BETWEEN:-" or a bold party
    name) does NOT open a new section — only a structural heading does. This
    stops the preamble from being fragmented across chunks, which made it
    impossible to retrieve as one block. The behaviour is disabled when the
    document has no structural headings at all, so structure-less documents are
    not collapsed into one giant chunk.
    """
    blocks = detect_headings(blocks)
    has_structural = any(
        b.is_heading and _is_structural_section_heading(b.text) for b in blocks
    )

    chunks: list[SectionChunk] = []
    current_heading = ""
    current_body: list[str] = []
    current_pages: set[int] = set()
    seen_structural = False

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
            is_structural = _is_structural_section_heading(block.text)

            # Preamble region: before the first structural heading, a non-
            # structural heading is folded into the running preamble chunk
            # rather than starting a new section.
            if has_structural and not seen_structural and not is_structural:
                if _is_preamble_opener(block.text):
                    # Operative opener ("THIS AGREEMENT is dated …") → start the
                    # preamble proper here so it anchors the chunk heading.
                    # Flush any preceding cover/title into its own chunk.
                    saved = flush(len(chunks))
                    if saved:
                        chunks.append(saved)
                    current_heading = block.text
                    current_body = []
                    current_pages = {block.page}
                elif not current_heading and not current_body:
                    # First block in the document → seed the preamble heading.
                    current_heading = block.text
                    current_pages = {block.page}
                else:
                    current_body.append(block.text)
                    current_pages.add(block.page)
                continue

            if is_structural:
                seen_structural = True

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


# ── Step 3b: Short chunk filter ───────────────────────────────────────────────

# Minimum number of characters the body content of a chunk must have.
# A chunk whose entire body is shorter than this is almost certainly an
# artefact: a stray bold label, a misdetected heading that captured nothing,
# or a navigation element that slipped through.
#
# Typical real sections have body content in the hundreds of characters.
# Setting this to 30 removes clear junk while preserving all real content,
# including very short clauses like "Not applicable." or "See Schedule 2."
_MIN_BODY_CHARS = 30


def filter_short_chunks(
    chunks: list[SectionChunk],
    min_body_chars: int = _MIN_BODY_CHARS,
) -> list[SectionChunk]:
    """
    Fold short, non-structural fragments back into their relevant section
    instead of discarding them, so NO document text is lost.

    By the time chunking runs, the only legitimate removals (TOC pages, running
    headers/footers, page numbers) have already happened upstream — so a short
    chunk left here is real prose that simply got over-split, most often the
    preamble/parties block:

        "THIS AGREEMENT is dated … and made BETWEEN:-"
        "(1) <corporate name>"          ← short fragment
        "(2) <bank name>"               ← short fragment

    Previously these short fragments were deleted, which is why such a section
    could not be retrieved. Now each short fragment's full text (heading + body)
    is appended to the PRECEDING kept chunk — its relevant section — preserving
    every word and keeping the section whole.

    A chunk is kept as its own section (not merged) when EITHER:
    - its full text (heading + body) is at least min_body_chars long, OR
    - it is a structural anchor — a numbered heading ('ARTICLE I', 'Section
      1.01', '§ 2.3') or a named section ('Schedule 2', 'Annex A') — even when
      its direct body is empty, since its content lives in child sections.

    The size test is on FULL text (heading + body): a bold defined term such as
    '"Margin" means 0.80 per cent.' lives in the heading with an empty body and
    must be kept on the strength of the heading alone.
    """
    kept: list[SectionChunk] = []
    merged_count = 0
    merged_samples: list[str] = []

    for chunk in chunks:
        total_chars = len(chunk.full_text.strip())   # heading + body
        # Structural anchors (US/UK): bare numbers ('1.0.1'), label-prefixed
        # ('Section 1.0.1'), roman ('ARTICLE I'), section symbol ('§ 2.3'),
        # and named sections ('Schedule 2', 'Annex A').
        is_structural = (
            _extract_section_number(chunk.heading) is not None
            or _extract_named_prefix(chunk.heading) is not None
        )

        if total_chars >= min_body_chars or is_structural:
            kept.append(chunk)
            continue

        # Short, non-structural fragment → merge its text into the relevant
        # (preceding) section rather than dropping it.
        fragment = chunk.full_text.strip()
        if not fragment:
            continue  # genuinely empty after stripping — nothing to preserve

        if kept:
            prev = kept[-1]
            new_content = f"{prev.content}\n{fragment}".strip() if prev.content else fragment
            new_pages = sorted(set(prev.pages) | set(chunk.pages))
            kept[-1] = pydantic_copy(prev, {"content": new_content, "pages": new_pages})
        else:
            # No preceding section yet (document opens with a short fragment):
            # keep it so following fragments can merge into it.
            kept.append(chunk)
            continue

        merged_count += 1
        if len(merged_samples) < 5:
            merged_samples.append(repr(chunk.heading[:40]))

    if merged_count:
        print(f"      → Merged {merged_count} short fragment(s) into their "
              f"section instead of dropping: {merged_samples}")

    return kept


# ── Step 4: Token-safe splitting ──────────────────────────────────────────────

def split_oversized_chunks(
    chunks: list[SectionChunk],
    max_tokens: int = TARGET_CHUNK_TOKENS,
    chunk_overlap: int = BGE_M3_CHUNK_OVERLAP,
) -> list[SectionChunk]:
    """
    Split any SectionChunk whose full_text exceeds `max_tokens` into sub-chunks,
    using the real BGE-M3 tokeniser as the length function.  Heading and page
    metadata are propagated to every sub-chunk; splits are labelled
    "(part 1)", "(part 2)", … and reassembled by gather_full_section for
    extract mode.

    `max_tokens` defaults to TARGET_CHUNK_TOKENS (a retrieval-quality target),
    NOT the 8192 embedding ceiling — see TARGET_CHUNK_TOKENS for why smaller
    chunks retrieve better.  It is clamped to the embedding limit for safety.
    """
    max_tokens = min(max_tokens, BGE_M3_MAX_TOKENS)

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
        target_chunk_tokens: int = TARGET_CHUNK_TOKENS,
    ) -> "DocumentIndex":
        """
        Run the five shared preparation phases and return a DocumentIndex.
        This is the only step that reads the PDF — all subsequent operations
        work entirely from the returned index.

        target_chunk_tokens : retrieval-quality target size for splitting large
                              sections (default TARGET_CHUNK_TOKENS=1000).  Lower
                              it (e.g. 600) for finer-grained retrieval on dense
                              US agreements; raise it to keep larger sections
                              whole.  Never exceeds the 8192 embedding ceiling.
        """
        print(f"[1/5] Extracting blocks from: {pdf_path}")
        blocks = extract_structured_blocks(pdf_path)
        print(f"      → {len(blocks)} blocks extracted")

        print("[2/5] Filtering TOC pages & repeating artefacts...")
        blocks = filter_toc_blocks(
            blocks, llm_model=llm_model, base_url=base_url, api_key=api_key,
        )
        blocks = filter_repeating_blocks(blocks)
        blocks = filter_positional_headers_footers(blocks)
        print(f"      → {len(blocks)} blocks after filtering")

        print("[3/5] Chunking by section...")
        blocks = split_structural_headings(blocks)
        chunks = chunk_by_section(blocks)
        print(f"      → {len(chunks)} raw sections found")
        chunks = filter_short_chunks(chunks)
        print(f"      → {len(chunks)} sections after folding short fragments "
              f"into their relevant section")

        print("[4/5] Splitting oversized chunks...")
        chunks = split_oversized_chunks(chunks, max_tokens=target_chunk_tokens)
        print(f"      → {len(chunks)} chunks after token-safe split "
              f"(target {min(target_chunk_tokens, BGE_M3_MAX_TOKENS)} tokens)")

        print("[5/5] Building vector store...")
        vector_store = build_vector_store(
            chunks, base_url=base_url, api_key=api_key, model=embedding_model,
        )
        print(f"      → Vector store ready ({len(chunks)} documents indexed)")

        return cls(pdf_path=pdf_path, chunks=chunks, vector_store=vector_store)
