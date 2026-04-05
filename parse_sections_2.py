"""
PDF Section Parser v2 — Heading Keys
======================================
Same parsing logic as parse_sections.py, but keys include the section
heading text so the dictionary is self-describing and searchable by
topic without inspecting values.

Key format:
    parse_sections.py  →  {"1": "...", "1.1": "...", "3.3": "..."}
    parse_sections_2.py→  {"1 DEFINITIONS AND INTERPRETATION": "...",
                           "1.1 Definitions": "...",
                           "3.3 Financial Covenants": "..."}

Sections at depth ≥ 3 ("3.3.1", "3.3.2") are still folded into their
depth-2 parent key, exactly as in parse_sections.py.

Usage:
    python parse_sections_2.py input.pdf
    python parse_sections_2.py input.pdf --output sections.json
    python parse_sections_2.py input.pdf --show-keys
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Re-use shared extraction utilities — no duplication
from parse_sections import (
    MAX_DEPTH,
    _SECTION_RE,
    _normalise_key,
    _section_depth,
    extract_blocks,
)


# ─────────────────────────────────────────────
# Heading key builder
# ─────────────────────────────────────────────

def _build_heading_key(number_key: str, first_line: str) -> str:
    """
    Combine the normalised section number with the heading text from the
    first line of a block.

    Examples
    --------
    "1"   + "1. DEFINITIONS AND INTERPRETATION"  ->  "1 DEFINITIONS AND INTERPRETATION"
    "1.1" + "1.1 Definitions"                    ->  "1.1 Definitions"
    "3.3" + "3.3 Financial Covenants"            ->  "3.3 Financial Covenants"
    """
    # Strip the leading section number (and its optional trailing dot) from the line
    heading_text = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', first_line).strip()
    if heading_text:
        return f"{number_key} {heading_text}"
    return number_key


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_sections(blocks: list[str]) -> dict[str, str]:
    """
    Walk blocks in order and build a heading-key → text mapping.

    Rules
    -----
    - depth <= MAX_DEPTH : start a new key  "{number} {heading}"
    - depth > MAX_DEPTH  : fold into the current key as body text
    - Text before the first heading lands under the key "preamble"
    """
    sections: dict[str, str] = {}
    current_heading_key: str | None = None   # the dict key in use
    current_number_key:  str | None = None   # numeric key, needed for depth-3 fallback
    current_lines: list[str] = []

    def _flush() -> None:
        if current_heading_key is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                if current_heading_key in sections:
                    sections[current_heading_key] = sections[current_heading_key] + "\n" + text
                else:
                    sections[current_heading_key] = text

    for block in blocks:
        first_line = block.splitlines()[0].strip()
        match = _SECTION_RE.match(first_line)

        if match:
            raw_key    = match.group(1)
            number_key = _normalise_key(raw_key)
            depth      = _section_depth(number_key)

            if depth <= MAX_DEPTH:
                _flush()
                current_number_key  = number_key
                current_heading_key = _build_heading_key(number_key, first_line)
                current_lines = [block]
            else:
                # depth 3+: fold as body text under current key
                if current_heading_key is None:
                    # Edge case: deep section before any shallow one
                    parent_number = ".".join(number_key.split(".")[:MAX_DEPTH])
                    current_number_key  = parent_number
                    current_heading_key = parent_number   # no heading available yet
                    current_lines = [block]
                else:
                    current_lines.append(block)
        else:
            if current_heading_key is None:
                current_heading_key = "preamble"
                current_number_key  = "preamble"
            current_lines.append(block)

    _flush()
    return sections


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse a PDF into a heading-key dictionary."
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

    blocks   = extract_blocks(str(pdf_path))
    sections = parse_sections(blocks)

    if args.show_keys:
        for key in sections:
            print(key)
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
