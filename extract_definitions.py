"""
Definitions Extractor
=====================
Locates the "Definitions and Interpretation" section in a facility agreement
PDF and parses it into a structured JSON dictionary.

Each entry in the output maps a defined term to its full definition text:

    {
      "Availability Period": "\\"Availability Period\\" means the period
                              commencing on the date of this Agreement...",
      "Business Day": "\\"Business Day\\" means a day (other than a Saturday
                       or Sunday) on which banks are open..."
    }

Pipeline
--------
1. extract_all_blocks(pdf_path)
       Opens the PDF with PyMuPDF and returns every text block in reading
       order across all pages.  Image blocks are discarded.

2. find_definitions_section(blocks)
       Scans blocks for the first heading that contains
       "Definitions and Interpret..." (case-insensitive).  Collects all
       subsequent blocks until the next depth-1 numbered section heading
       (e.g. "2. THE FACILITY") signals the end of the section.

3. parse_definitions(blocks)
       Walks the collected blocks.  A new definition entry begins whenever a
       block's first line opens with a quoted term (straight or typographic
       quotes: " " ' ' « »).  Subsequent blocks that do not start with a
       quoted term are treated as continuations and folded into the current
       entry.  The key is the bare term (no quotes); the value is the full
       normalised definition text.

Usage:
    python extract_definitions.py input.pdf
    python extract_definitions.py input.pdf --output definitions.json
    python extract_definitions.py input.pdf --show-keys
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

# Matches the definitions section heading anywhere in a line.
# Handles: "1. DEFINITIONS AND INTERPRETATION"
#          "1.1 Definitions"
#          "Definitions and Interpretations"
#          "DEFINITIONS AND INTERPRETATION"
_DEF_SECTION_RE = re.compile(
    r'definitions?\s+and\s+interpret|\bdefinitions?\s*$',
    re.IGNORECASE,
)

# Matches a depth-1 section heading at the start of a line:
# "2. " "10. " etc.  Does NOT match "1.1 " (depth-2) because the digit
# after the first dot is immediately followed by a space only for depth-1.
_DEPTH1_SECTION_RE = re.compile(r'^\d+\.\s')

# Matches a depth-2 section heading at the start of a line:
# "1.1 " "1.2 " "10.3 " etc.
_DEPTH2_SECTION_RE = re.compile(r'^\d+\.\d+')

# Matches a quoted term at the very start of a block's first line.
# Handles straight quotes (") and typographic/smart quotes (" " ' ' « »).
# Group 1 → the bare term text, e.g. "Availability Period"
_OPEN_QUOTES  = r'["\u201c\u2018\u00ab]'
_CLOSE_QUOTES = r'["\u201d\u2019\u00bb]'
_DEF_TERM_RE  = re.compile(
    rf'^{_OPEN_QUOTES}([A-Z][^"\u201c\u201d\u2018\u2019\u00ab\u00bb]+?){_CLOSE_QUOTES}',
    re.UNICODE,
)


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_all_blocks(pdf_path: str) -> list[str]:
    """
    Return every text block from the PDF in reading order.

    Image blocks are discarded.  All pages are included.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    list[str]
        Non-empty text blocks in document order.
    """
    blocks: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for block in page.get_text("blocks"):
                # block layout: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:          # skip image blocks
                    continue
                text = block[4].strip()
                if text:
                    blocks.append(text)
    return blocks


# ─────────────────────────────────────────────
# Section finder
# ─────────────────────────────────────────────

def find_definitions_section(blocks: list[str]) -> list[str]:
    """
    Locate and return the blocks that belong to the definitions section.

    The section begins on the block AFTER the first heading that matches
    ``_DEF_SECTION_RE`` (covers "1. Definitions and Interpretation",
    "1.1 Definitions", bare "Definitions", etc.).  It ends immediately
    before the next section heading at the same depth or higher — depth-1
    (e.g. "2. THE FACILITY") always stops; depth-2 (e.g. "1.2 Interpretation")
    stops only when the definitions heading was itself depth-2.

    Parameters
    ----------
    blocks:
        All text blocks from the PDF (output of ``extract_all_blocks``).

    Returns
    -------
    list[str]
        Blocks that form the body of the definitions section, excluding the
        heading line itself.  Returns an empty list if the section is not
        found.
    """
    start_idx:        int | None = None
    heading_is_depth2: bool      = False

    for i, block in enumerate(blocks):
        first_line = block.splitlines()[0].strip()

        if start_idx is None:
            # Look for the definitions heading
            if _DEF_SECTION_RE.search(first_line):
                start_idx        = i + 1       # content starts after the heading
                heading_is_depth2 = bool(_DEPTH2_SECTION_RE.match(first_line))
        else:
            # Stop at the next section heading that is NOT the definitions
            # section itself.  When the definitions heading was depth-2
            # (e.g. "1.1 Definitions"), also stop at sibling depth-2 headings
            # (e.g. "1.2 Interpretation").  When it was depth-1, only depth-1
            # headings terminate the section (existing behaviour).
            is_depth1 = bool(_DEPTH1_SECTION_RE.match(first_line))
            is_depth2 = bool(_DEPTH2_SECTION_RE.match(first_line))
            is_section = (is_depth1 or (heading_is_depth2 and is_depth2))
            if is_section and not _DEF_SECTION_RE.search(first_line):
                return blocks[start_idx:i]

    if start_idx is None:
        return []

    # Reached end of document without finding a successor section
    return blocks[start_idx:]


# ─────────────────────────────────────────────
# Definition parser
# ─────────────────────────────────────────────

def parse_definitions(blocks: list[str]) -> dict[str, str]:
    """
    Parse definition blocks into a ``{term: definition_text}`` dictionary.

    Rules
    -----
    - A block whose first line begins with a quoted term (``_DEF_TERM_RE``)
      starts a new definition entry.
    - Subsequent blocks that do NOT start with a quoted term are continuations
      of the current entry and are folded in.
    - Blocks before the first quoted term are discarded (e.g. preamble text
      inside the definitions section).

    Text normalisation: whitespace within each block is collapsed to single
    spaces; blocks belonging to the same definition are joined with a space.

    Parameters
    ----------
    blocks:
        Blocks from the definitions section (output of
        ``find_definitions_section``).

    Returns
    -------
    dict[str, str]
        ``{bare_term: full_definition_text}``
        e.g. ``{"Availability Period": '"Availability Period" means...'}``
    """
    definitions:   dict[str, str] = {}
    current_term:  str | None     = None
    current_parts: list[str]      = []

    def _flush() -> None:
        if current_term and current_parts:
            # Normalise intra-block whitespace; join blocks with a space
            full_text = " ".join(
                " ".join(part.split()) for part in current_parts
            ).strip()
            if full_text:
                definitions[current_term] = full_text

    for block in blocks:
        first_line = block.splitlines()[0].strip()
        match = _DEF_TERM_RE.match(first_line)

        if match:
            _flush()
            current_term  = match.group(1).strip()
            current_parts = [block]
        else:
            if current_term is not None:
                current_parts.append(block)
            # blocks before the first quoted term are ignored

    _flush()
    return definitions


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the Definitions and Interpretation section from a "
            "facility agreement PDF and output it as a JSON dictionary."
        )
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
        help="Print only the extracted term keys, not the full definitions",
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    blocks      = extract_all_blocks(str(pdf_path))
    def_blocks  = find_definitions_section(blocks)

    if not def_blocks:
        print(
            "Warning: 'Definitions and Interpretation' section not found.",
            file=sys.stderr,
        )
        sys.exit(1)

    definitions = parse_definitions(def_blocks)

    if not definitions:
        print(
            "Warning: section found but no definitions could be parsed.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.show_keys:
        for term in definitions:
            print(term)
        return

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(definitions, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Saved {len(definitions)} definitions to {out_path}")
    else:
        print(json.dumps(definitions, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
