"""
PDF Schedule Parser
===================
Parses a PDF into a dictionary keyed by schedule name.

Each "Schedule N" heading (case-insensitive) starts a new entry.  All text
that follows — until the next schedule heading or end of document — is
collected as the value for that key.

Key format:
    "Schedule 1": "..."
    "Schedule 2": "..."

Usage:
    python parse_schedule.py input.pdf
    python parse_schedule.py input.pdf --output schedules.json
    python parse_schedule.py input.pdf --show-keys
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Re-use the shared PDF extraction utility — no duplication
from parse_sections import extract_blocks


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Matches a schedule heading at the very start of a line, e.g.:
#   "Schedule 1"  "SCHEDULE 1"  "Schedule 1 – Conditions Precedent"
# Group 1 → schedule number (digits only)
# Group 2 → optional title text on the same line (may be empty)
_SCHEDULE_RE = re.compile(r'^schedule\s+(\d+)\b\s*(.*)', re.IGNORECASE)


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_schedules(blocks: list[str]) -> dict[str, str]:
    """
    Walk blocks in order and build a Schedule-key → text mapping.

    Rules
    -----
    - A block whose first line matches _SCHEDULE_RE starts a new schedule.
    - All subsequent blocks belong to that schedule until the next one begins.
    - Blocks before the first schedule heading are discarded (they belong to
      the main agreement body, not to any schedule).
    """
    schedules: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

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
        match = _SCHEDULE_RE.match(first_line)

        if match:
            _flush()
            number = match.group(1)
            current_key = f"Schedule {number}"
            current_lines = [block]
        else:
            if current_key is not None:
                current_lines.append(block)
            # blocks before any schedule heading are ignored

    _flush()
    return schedules


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse schedules from a PDF into a keyed dictionary."
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

    blocks    = extract_blocks(str(pdf_path))
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
