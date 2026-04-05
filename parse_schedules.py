"""
PDF Schedules Parser
====================
Parses schedules from the second half of a PDF into a dictionary keyed by
schedule name.

Two-stage pipeline
------------------
1. extract_blocks_second_half
       Opens the PDF, computes the midpoint page, and returns text blocks
       from that page onwards.  Schedules always appear at the tail of a
       facility agreement; restricting extraction to the second half avoids
       false-positive "Schedule N" matches from in-body cross-references
       (e.g. "as set out in Schedule 1") in the main clauses.

2. parse_schedules
       Walks the blocks and collects content under each "Schedule N" heading.
       Collection stops when a block's first line matches
       "IN WITNESS WHEREOF" (case-insensitive), preventing the last schedule
       from absorbing the execution / signature section.

Key format:
    "Schedule 1": "..."
    "Schedule 2": "..."

Usage:
    python parse_schedules.py input.pdf
    python parse_schedules.py input.pdf --output schedules.json
    python parse_schedules.py input.pdf --show-keys
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import pymupdf


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Matches a schedule heading at the start of a line (case-insensitive):
#   "Schedule 1"  "SCHEDULE 1"  "Schedule 1 – Conditions Precedent"
# Group 1 → schedule number
# Group 2 → optional title text on the same line
_SCHEDULE_RE = re.compile(r'^schedule\s+(\d+)\b\s*(.*)', re.IGNORECASE)

# Stops collection when the execution clause begins.
_STOP_RE = re.compile(r'^in\s+witness\s+whereof', re.IGNORECASE)


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_blocks_second_half(pdf_path: str) -> list[str]:
    """
    Return text blocks from the second half of the PDF in reading order.

    The document is split at ``ceil(total_pages / 2)``.  All pages from that
    index to the end are included; the first half is skipped entirely.
    Image blocks are discarded.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    list[str]
        Non-empty text blocks from the second half of the document.
    """
    blocks: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
        start_page  = math.ceil(total_pages / 2)   # first page of the second half

        for page_index in range(start_page, total_pages):
            page = doc[page_index]
            for block in page.get_text("blocks"):
                # block layout: (x0, y0, x1, y1, text, block_no, block_type)
                if block[6] != 0:       # skip image blocks
                    continue
                text = block[4].strip()
                if text:
                    blocks.append(text)

    return blocks


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_schedules(blocks: list[str]) -> dict[str, str]:
    """
    Walk blocks in order and build a Schedule-key → text mapping.

    Rules
    -----
    - A block whose first line matches ``_SCHEDULE_RE`` starts a new schedule.
    - All subsequent blocks belong to that schedule until:
        (a) the next schedule heading appears, OR
        (b) a block's first line matches ``_STOP_RE``
            ("IN WITNESS WHEREOF") — the current schedule is flushed and
            collection halts.
    - Blocks before the first schedule heading are discarded.

    Parameters
    ----------
    blocks:
        Text blocks as returned by ``extract_blocks_second_half``.

    Returns
    -------
    dict[str, str]
        Mapping of ``"Schedule N"`` → full schedule text.
    """
    schedules:     dict[str, str] = {}
    current_key:   str | None     = None
    current_lines: list[str]      = []

    def _flush() -> None:
        if current_key is not None and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                if current_key in schedules:
                    schedules[current_key] = schedules[current_key] + "\n" + text
                else:
                    schedules[current_key] = text

    for block in blocks:
        first_line = block.splitlines()[0].strip()

        # Stop on execution clause — flush and exit
        if current_key is not None and _STOP_RE.match(first_line):
            _flush()
            break

        sched_match = _SCHEDULE_RE.match(first_line)
        if sched_match:
            _flush()
            current_key   = f"Schedule {sched_match.group(1)}"
            current_lines = [block]
        else:
            if current_key is not None:
                current_lines.append(block)
            # blocks before the first schedule heading are ignored

    _flush()
    return schedules


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse schedules from the second half of a PDF, "
            "stopping at 'IN WITNESS WHEREOF'."
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
        help="Print only the detected schedule keys, not the full text",
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    blocks    = extract_blocks_second_half(str(pdf_path))
    schedules = parse_schedules(blocks)

    if args.show_keys:
        for key in schedules:
            print(key)
        return

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(
            json.dumps(schedules, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Saved {len(schedules)} schedules to {out_path}")
    else:
        print(json.dumps(schedules, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
