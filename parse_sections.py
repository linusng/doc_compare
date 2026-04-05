"""
PDF Section Parser
==================
Parses a PDF into a dictionary keyed by section number (up to depth 2).

Sections at depth 1 ("1.", "2.") and depth 2 ("1.1", "2.3") each become
a separate key. Deeper sections ("3.3.1", "3.3.2") are kept as body text
under their parent depth-2 key rather than split out.

A block is only treated as a section heading if its leading number contains
at least one period — either a trailing dot ("2.", "14.") or an internal dot
("1.1", "3.3.2"). Bare integers ("42") are never promoted to section keys,
so page numbers embedded in the text are silently folded into the current
section's body rather than starting a new one.

Usage:
    python parse_sections.py input.pdf
    python parse_sections.py input.pdf --output sections.json
    python parse_sections.py input.pdf --show-keys
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

# Matches section numbers at the very start of a line.
# A valid section number MUST contain at least one period — either:
#   • a trailing dot:   "1."  "14."
#   • internal dot(s):  "1.1"  "3.3.2"
# Bare integers like "42" (page numbers) are intentionally excluded.
# Group 1 → raw section number (e.g. "1.", "1.1", "3.3.1")
# Group 2 → heading text on the same line (may be empty for stand-alone number lines)
_SECTION_RE = re.compile(r'^(\d+(?:\.\d+)+\.?|\d+\.)\s*(.*)')

MAX_DEPTH = 2


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

def extract_blocks(pdf_path: str) -> list[str]:
    """Return content text blocks from the PDF in reading order.

    Image blocks are skipped; all text blocks are returned as-is.
    Page numbers and other bare integers in the text are handled at the
    parsing stage: ``_SECTION_RE`` requires a period so they are never
    promoted to section keys.
    """
    blocks: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            for block in page.get_text("blocks"):
                # block layout: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:           # skip image blocks
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
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    blocks = extract_blocks(str(pdf_path))
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
