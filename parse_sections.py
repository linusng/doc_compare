"""
PDF Section Parser
==================
Parses a PDF into a dictionary keyed by section number (up to depth 2).

Sections at depth 1 ("1.", "2.") and depth 2 ("1.1", "2.3") each become
a separate key. Deeper sections ("3.3.1", "3.3.2") are kept as body text
under their parent depth-2 key rather than split out.

Running headers and footers (page numbers, document title, confidentiality
notices) are excluded by filtering any block whose bounding box lies entirely
within the top or bottom margin strip of its page.

Usage:
    python parse_sections.py input.pdf
    python parse_sections.py input.pdf --output sections.json
    python parse_sections.py input.pdf --show-keys
    python parse_sections.py input.pdf --header-margin 60 --footer-margin 60
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

# Matches section numbers at the very start of a line:
#   "1."  "1.1"  "1.1.1"  "3.3.2"  (trailing dot optional)
# Group 1 → raw section number (e.g. "1.", "1.1", "3.3.1")
# Group 2 → heading text on the same line (may be empty for stand-alone number lines)
_SECTION_RE = re.compile(r'^(\d+(?:\.\d+)*)\.?\s*(.*)')

MAX_DEPTH = 2

# Default margin strips (points) excluded from each page edge.
# Standard A4/Letter headers and footers live within ~50 pt of the page edge.
# Tune with --header-margin / --footer-margin if your document differs.
_HEADER_MARGIN: float = 50.0
_FOOTER_MARGIN: float = 50.0


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _section_depth(key: str) -> int:
    """Return the nesting depth of a normalised section key.

    "1"     → 1
    "1.1"   → 2
    "1.1.1" → 3
    """
    return len(key.split("."))


def _normalise_key(raw: str) -> str:
    """Strip any trailing dot so '1.' and '1' both become '1'."""
    return raw.rstrip(".")


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_blocks(
    pdf_path: str,
    header_margin: float = _HEADER_MARGIN,
    footer_margin: float = _FOOTER_MARGIN,
) -> list[str]:
    """Return content text blocks from the PDF in reading order.

    Blocks whose bounding box lies entirely within the top ``header_margin``
    points or the bottom ``footer_margin`` points of their page are skipped.
    This removes running headers, running footers, and page numbers without
    touching body content.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.
    header_margin:
        Height in points of the top strip to exclude (default 50 pt ≈ 17 mm).
    footer_margin:
        Height in points of the bottom strip to exclude (default 50 pt ≈ 17 mm).
    """
    blocks: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            page_height = page.rect.height
            footer_y    = page_height - footer_margin   # y-coordinate where footer zone begins

            for block in page.get_text("blocks"):
                # block layout: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:           # skip image blocks
                    continue

                y0, y1 = block[1], block[3]

                # Skip if the block sits entirely within the header strip
                if y1 <= header_margin:
                    continue

                # Skip if the block sits entirely within the footer strip
                if y0 >= footer_y:
                    continue

                text = block[4].strip()
                if text:
                    blocks.append(text)

    return blocks


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_sections(blocks: list[str]) -> dict[str, str]:
    """
    Walk blocks in order and build a section → text mapping.

    Rules
    -----
    - A block whose first line matches _SECTION_RE is a candidate heading.
    - If that section's depth <= MAX_DEPTH  → start a new key.
    - If depth > MAX_DEPTH                  → fold into the current key as body text.
    - Text before the first heading lands under the synthetic key "preamble".
    """
    sections: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_key is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                if current_key in sections:
                    sections[current_key] = sections[current_key] + "\n" + text
                else:
                    sections[current_key] = text

    for block in blocks:
        first_line = block.splitlines()[0].strip()
        match = _SECTION_RE.match(first_line)

        if match:
            raw_key = match.group(1)
            key = _normalise_key(raw_key)
            depth = _section_depth(key)

            if depth <= MAX_DEPTH:
                # New section: flush previous, start fresh
                _flush()
                current_key = key
                current_lines = [block]
            else:
                # Deep section (3+): fold as body text under current key
                if current_key is None:
                    # Edge case: deep section before any shallow one
                    current_key = ".".join(key.split(".")[:MAX_DEPTH])
                    current_lines = [block]
                else:
                    current_lines.append(block)
        else:
            # Plain body text
            if current_key is None:
                current_key = "preamble"
            current_lines.append(block)

    _flush()
    return sections


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PDF into a section-keyed dictionary."
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
        help="Print only the detected section keys, not the full text",
    )
    parser.add_argument(
        "--header-margin",
        type=float,
        default=_HEADER_MARGIN,
        help=f"Points to exclude from the top of each page (default: {_HEADER_MARGIN})",
    )
    parser.add_argument(
        "--footer-margin",
        type=float,
        default=_FOOTER_MARGIN,
        help=f"Points to exclude from the bottom of each page (default: {_FOOTER_MARGIN})",
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    blocks = extract_blocks(
        str(pdf_path),
        header_margin=args.header_margin,
        footer_margin=args.footer_margin,
    )
    sections = parse_sections(blocks)

    if args.show_keys:
        for key in sections:
            print(key)
        return

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(sections, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved {len(sections)} sections to {out_path}")
    else:
        print(json.dumps(sections, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
