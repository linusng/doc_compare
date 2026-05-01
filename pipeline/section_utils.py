"""
Heading and section utilities shared by both extract and collate modes.

Covers:
- Heading normalisation (_base_heading, _extract_section_number)
- Section hierarchy helpers (_is_child_section)
- Named structural section helpers (Schedule N, Annex A, …)
- Chunk ↔ LangChain Document conversion
- Full-section gathering (gather_full_section, merge_section_chunks)
"""

import re

from langchain_core.documents import Document

from .models import ExtractionResult, SectionChunk


# ── Heading normalisation ─────────────────────────────────────────────────────

def _base_heading(heading: str) -> str:
    """
    Strip '(part N)' suffixes that split_oversized_chunks adds, so split
    fragments can be matched back to their parent section.

    '1.1 Definitions (part 2)' → '1.1 Definitions'
    """
    return re.sub(r'\s*\(part \d+\)\s*$', '', heading).strip()


def _extract_section_number(heading: str) -> str | None:
    """
    Return the leading section number from a heading.

    '1.1 Definitions' → '1.1'
    '19. Extension'   → '19'
    'Schedule 2 ...'  → None  (handled by named-prefix logic instead)
    """
    heading = _base_heading(heading)
    m = re.match(r'^(\d+(?:\.\d+)*)\s*\.?\s', heading.strip())
    return m.group(1) if m else None


# ── Section hierarchy ─────────────────────────────────────────────────────────

def _is_child_section(parent_num: str, candidate_heading: str) -> bool:
    """
    True if candidate_heading is a sub-section of parent_num.

    parent_num='1'  → '1.1', '1.2', '1.15' are children; '2', '2.1' are not.
    Also matches split parts of the same heading (same section number).
    """
    cand_num = _extract_section_number(candidate_heading)
    if cand_num is None:
        return False
    if cand_num == parent_num:
        return True
    return cand_num.startswith(parent_num + ".")


# ── Named structural sections (Schedule N, Annex A, Appendix 3, …) ────────────

_NAMED_PREFIX_RE = re.compile(
    r'^(schedule|annex|appendix|exhibit|attachment|part)\s+(\w+)',
    re.IGNORECASE,
)


def _extract_named_prefix(heading: str) -> tuple[str, str] | None:
    """
    For 'Schedule 2 Compounded Rate Terms' → ('schedule', '2').
    For 'Annex A Definitions'              → ('annex', 'a').
    Returns None if the heading doesn't follow this pattern.
    """
    m = _NAMED_PREFIX_RE.match(_base_heading(heading).strip())
    return (m.group(1).lower(), m.group(2).lower()) if m else None


def _is_sibling_named_section(best_heading: str, candidate_heading: str) -> bool:
    """
    True when candidate_heading is a peer of best_heading under the same
    structural type but a different identifier.

    best='Schedule 2 …', candidate='Schedule 3 …' → True  (stop collecting)
    best='Schedule 2 …', candidate='Part 1 …'     → False (different type)
    """
    best_prefix = _extract_named_prefix(best_heading)
    cand_prefix = _extract_named_prefix(candidate_heading)
    if best_prefix is None or cand_prefix is None:
        return False
    return best_prefix[0] == cand_prefix[0] and best_prefix[1] != cand_prefix[1]


# ── Chunk ↔ Document conversion ───────────────────────────────────────────────

def chunk_to_doc(chunk: SectionChunk) -> Document:
    """Convert a SectionChunk to a LangChain Document with full metadata."""
    return Document(
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


# ── Full-section gathering ────────────────────────────────────────────────────

def gather_full_section(
    best_doc: Document,
    all_chunks: list[SectionChunk],
) -> list[SectionChunk]:
    """
    Given the best-matched chunk, collect ALL chunks that belong to that
    exact section — its split parts and any non-numbered body text that
    follows. Does NOT collect sibling sections.

    Handles three heading types:

    1. Numbered sections (e.g. '1.1 Definitions'):
       Walks forward collecting child sections (1.1.1, 1.1.2, …) and
       non-numbered bold blocks. Stops at the next sibling (1.2, 2, …).

    2. Named structural sections (e.g. 'Schedule 2 Compounded Rate Terms'):
       Walks forward collecting everything until the next sibling schedule
       (Schedule 3, …).

    3. Unstructured headings (bold text with no number or named prefix):
       Returns only split parts of the same base heading.
    """
    best_id = best_doc.metadata["chunk_id"]
    best_heading = best_doc.metadata["heading"]
    best_base = _base_heading(best_heading)
    best_num = _extract_section_number(best_heading)

    start_idx = next(
        (i for i, c in enumerate(all_chunks) if c.chunk_id == best_id), None
    )
    if start_idx is None:
        return [c for c in all_chunks if c.chunk_id == best_id]

    # ── Named structural section (Schedule N, Annex A, …) ────────────────────
    if best_num is None:
        if _extract_named_prefix(best_heading) is None:
            # Truly unstructured — only return split parts of this chunk.
            return [c for c in all_chunks if _base_heading(c.heading) == best_base]

        collected: list[SectionChunk] = []
        for i in range(start_idx, len(all_chunks)):
            chunk = all_chunks[i]
            if i == start_idx:
                collected.append(chunk)
                continue
            if _is_sibling_named_section(best_heading, chunk.heading):
                break
            collected.append(chunk)

        # Walk backward for any earlier split parts of the same heading.
        for i in range(start_idx - 1, -1, -1):
            chunk = all_chunks[i]
            if _base_heading(chunk.heading) == best_base:
                collected.insert(0, chunk)
            else:
                break

        return collected

    # ── Numbered section ──────────────────────────────────────────────────────
    collected = []
    for i in range(start_idx, len(all_chunks)):
        chunk = all_chunks[i]
        chunk_base = _base_heading(chunk.heading)
        cand_num = _extract_section_number(chunk.heading)

        if i == start_idx:
            collected.append(chunk)
            continue
        if chunk_base == best_base:          # split part of the same section
            collected.append(chunk)
            continue
        if cand_num is None:                 # non-numbered bold block inside section
            collected.append(chunk)
            continue
        if _is_child_section(best_num, chunk.heading):
            collected.append(chunk)
            continue
        break                                # sibling or unrelated section

    # Walk backward for any earlier split parts.
    for i in range(start_idx - 1, -1, -1):
        if _base_heading(all_chunks[i].heading) == best_base:
            collected.insert(0, all_chunks[i])
        else:
            break

    return collected


def merge_section_chunks(chunks: list[SectionChunk]) -> ExtractionResult:
    """
    Concatenate a list of SectionChunks into a single ExtractionResult,
    preserving sub-section headings and unioning all page numbers.
    """
    if not chunks:
        raise ValueError("No chunks to merge.")

    all_pages: set[int] = set()
    parts: list[str] = []

    for chunk in chunks:
        all_pages.update(chunk.pages)
        if chunk.heading and "(part " not in chunk.heading:
            parts.append(f"\n{chunk.heading}\n{chunk.content}")
        else:
            parts.append(chunk.content)

    return ExtractionResult(
        heading=_base_heading(chunks[0].heading),
        content="\n".join(parts).strip(),
        pages=sorted(all_pages),
        chunk_id=chunks[0].chunk_id,
    )
