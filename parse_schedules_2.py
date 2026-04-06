"""
PDF Schedules Parser v2
=======================
Same pipeline as parse_schedules.py but the output preserves the page
number(s) of every block that contributed to each schedule, enabling
precise traceability back to the source document.

Output format
-------------
Each schedule value is a nested dict rather than a plain string:

    {
      "Schedule 1": {
          "text": "Full schedule text...",
          "page_numbers": [14, 15, 16]
      },
      "Schedule 2": {
          "text": "...",
          "page_numbers": [17, 18]
      }
    }

``page_numbers`` is a sorted list of unique 1-based page numbers whose
text blocks contributed to that schedule.

Pipeline
--------
1. extract_blocks_second_half
       Opens the PDF, splits at ceil(total_pages / 2), and returns
       (text, page_number) pairs for every text block in the second half.

2. parse_schedules
       Walks the block pairs, groups content under "Schedule N" keys,
       and stops at "IN WITNESS WHEREOF" (case-insensitive).

Usage:
    python parse_schedules_2.py input.pdf
    python parse_schedules_2.py input.pdf --output schedules.json
    python parse_schedules_2.py input.pdf --show-keys
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

_SCHEDULE_RE = re.compile(r'^schedule\s+(\d+)\b\s*(.*)', re.IGNORECASE)
_STOP_RE     = re.compile(r'^in\s+witness\s+whereof', re.IGNORECASE)


# ─────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────

def extract_blocks_second_half(pdf_path: str) -> list[tuple[str, int]]:
    """
    Return (text, page_number) pairs from the second half of the PDF.

    The document is split at ``ceil(total_pages / 2)``.  Pages from that
    index to the end are included; the first half is skipped to avoid
    false-positive schedule headings from in-body cross-references.
    Image blocks are discarded.

    Parameters
    ----------
    pdf_path:
        Path to the PDF file.

    Returns
    -------
    list[tuple[str, int]]
        Each element is ``(block_text, page_number)`` where ``page_number``
        is 1-based (page 1 = first page of the document).
    """
    result: list[tuple[str, int]] = []

    with pymupdf.open(pdf_path) as doc:
        total_pages = doc.page_count
        start_page  = math.ceil(total_pages / 2)

        for page_index in range(start_page, total_pages):
            page        = doc[page_index]
            page_number = page_index + 1          # convert to 1-based

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

def parse_schedules(
    blocks: list[tuple[str, int]],
) -> dict[str, dict]:
    """
    Walk (text, page_number) pairs and build a Schedule-key → data mapping.

    Each value in the returned dict contains:
        "text"         — full schedule text (blocks joined with newlines)
        "page_numbers" — sorted list of unique 1-based page numbers

    Rules
    -----
    - A block whose first line matches ``_SCHEDULE_RE`` starts a new schedule.
    - All subsequent blocks belong to that schedule until:
        (a) the next schedule heading appears, OR
        (b) a block's first line matches ``_STOP_RE``
            ("IN WITNESS WHEREOF") — current schedule is flushed and
            collection halts.
    - Blocks before the first schedule heading are discarded.

    Parameters
    ----------
    blocks:
        List of ``(text, page_number)`` pairs as returned by
        ``extract_blocks_second_half``.

    Returns
    -------
    dict[str, dict]
        ``{"Schedule N": {"text": ..., "page_numbers": [...]}, ...}``
    """
    schedules:    dict[str, dict] = {}
    current_key:  str | None      = None
    current_text: list[str]       = []
    current_pages: list[int]      = []

    def _flush() -> None:
        if current_key is not None and current_text:
            text = "\n".join(current_text).strip()
            if text:
                page_numbers = sorted(set(current_pages))
                if current_key in schedules:
                    # Append to existing entry (shouldn't normally happen,
                    # but guard against duplicate headings)
                    schedules[current_key]["text"] += "\n" + text
                    schedules[current_key]["page_numbers"] = sorted(
                        set(schedules[current_key]["page_numbers"] + page_numbers)
                    )
                else:
                    schedules[current_key] = {
                        "text":         text,
                        "page_numbers": page_numbers,
                    }

    for text, page_number in blocks:
        first_line = text.splitlines()[0].strip()

        # Stop on execution clause
        if current_key is not None and _STOP_RE.match(first_line):
            _flush()
            break

        sched_match = _SCHEDULE_RE.match(first_line)
        if sched_match:
            _flush()
            current_key   = f"Schedule {sched_match.group(1)}"
            current_text  = [text]
            current_pages = [page_number]
        else:
            if current_key is not None:
                current_text.append(text)
                current_pages.append(page_number)
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
            "stopping at 'IN WITNESS WHEREOF'. "
            "Output includes page numbers for traceability."
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
        help="Print only the detected schedule keys, not the full content",
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
            pages = schedules[key]["page_numbers"]
            print(f"{key}  (pages {pages[0]}–{pages[-1]})")
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
