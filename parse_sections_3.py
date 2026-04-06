"""
PDF Section Parser v3 — with page numbers
==========================================
Same parsing logic as parse_sections.py but the output preserves the
page number(s) of every block that contributed to each section, enabling
precise traceability back to the source document.

Output format
-------------
Each section value is a nested dict rather than a plain string:

    {
      "1": {
          "text": "1. DEFINITIONS AND INTERPRETATION\\n...",
          "page_numbers": [1, 2, 3]
      },
      "1.1": {
          "text": "1.1 Definitions\\n...",
          "page_numbers": [2]
      }
    }

``page_numbers`` is a sorted list of unique 1-based page numbers whose
text blocks contributed to that section.

Parsing rules (unchanged from parse_sections.py)
-------------------------------------------------
- A section number must contain at least one period ("1.", "1.1", "3.3.2")
  so that bare integers (page numbers) are never promoted to section keys.
- Depth <= 2  → start a new key.
- Depth >= 3  → fold as body text under the current depth-2 key.
- Blocks before the first heading land under the synthetic key "preamble".

Usage:
    python parse_sections_3.py input.pdf
    python parse_sections_3.py input.pdf --output sections.json
    python parse_sections_3.py input.pdf --show-keys
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pymupdf


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Requires at least one period — bare integers never match.
_SECTION_RE = re.compile(r'^(\d+(?:\.\d+)+\.?|\d+\.)\s*(.*)')

MAX_DEPTH = 2


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _section_depth(key: str) -> int:
    return len(key.split("."))


def _normalise_key(raw: str) -> str:
    return raw.rstrip(".")


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_blocks(pdf_path: str) -> list[tuple[str, int]]:
    """
    Return ``(text, page_number)`` pairs for every text block in the PDF.

    Blocks are returned in reading order across all pages.  Image blocks
    are discarded.  ``page_number`` is 1-based.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    list[tuple[str, int]]
        Each element is ``(block_text, page_number)``.
    """
    result: list[tuple[str, int]] = []

    with pymupdf.open(pdf_path) as doc:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1          # 1-based

            for block in page.get_text("blocks"):
                # block layout: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:                 # skip image blocks
                    continue
                text = block[4].strip()
                if text:
                    result.append((text, page_number))

    return result


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_sections(
    blocks: list[tuple[str, int]],
) -> dict[str, dict]:
    """
    Walk ``(text, page_number)`` pairs and build a section-key → data mapping.

    Each value in the returned dict contains:
        "text"         — full section text (blocks joined with newlines)
        "page_numbers" — sorted list of unique 1-based page numbers

    Parameters
    ----------
    blocks:
        List of ``(text, page_number)`` pairs as returned by
        ``extract_blocks``.

    Returns
    -------
    dict[str, dict]
        ``{"1": {"text": ..., "page_numbers": [...]}, "1.1": {...}, ...}``
    """
    sections:      dict[str, dict] = {}
    current_key:   str | None      = None
    current_text:  list[str]       = []
    current_pages: list[int]       = []

    def _flush() -> None:
        if current_key is not None and current_text:
            text = "\n".join(current_text).strip()
            if text:
                page_numbers = sorted(set(current_pages))
                if current_key in sections:
                    sections[current_key]["text"] += "\n" + text
                    sections[current_key]["page_numbers"] = sorted(
                        set(sections[current_key]["page_numbers"] + page_numbers)
                    )
                else:
                    sections[current_key] = {
                        "text":         text,
                        "page_numbers": page_numbers,
                    }

    for text, page_number in blocks:
        first_line = text.splitlines()[0].strip()
        match = _SECTION_RE.match(first_line)

        if match:
            raw_key = match.group(1)
            key     = _normalise_key(raw_key)
            depth   = _section_depth(key)

            if depth <= MAX_DEPTH:
                _flush()
                current_key   = key
                current_text  = [text]
                current_pages = [page_number]
            else:
                # depth 3+: fold into current key as body text
                if current_key is None:
                    current_key   = ".".join(key.split(".")[:MAX_DEPTH])
                    current_text  = [text]
                    current_pages = [page_number]
                else:
                    current_text.append(text)
                    current_pages.append(page_number)
        else:
            if current_key is None:
                current_key   = "preamble"
                current_text  = []
                current_pages = []
            current_text.append(text)
            current_pages.append(page_number)

    _flush()
    return sections


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PDF into a section-keyed dictionary with page numbers."
    )
    parser.add_argument("input", help="Path to the input PDF file")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Save result as JSON to this path (default: print to stdout)",
    )
    parser.add_argument(
        "--show-keys",
        action="store_true",
        help="Print only the detected section keys, not the full content",
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    blocks   = extract_blocks(str(pdf_path))
    sections = parse_sections(blocks)

    if args.show_keys:
        for key, data in sections.items():
            pages = data["page_numbers"]
            page_str = (
                f"page {pages[0]}"
                if len(pages) == 1
                else f"pages {pages[0]}–{pages[-1]}"
            )
            print(f"{key}  ({page_str})")
        return

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {len(sections)} sections to {out_path}")
    else:
        print(json.dumps(sections, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
