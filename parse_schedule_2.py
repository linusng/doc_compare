"""
PDF Schedule Parser v2
=======================
Same as parse_schedule.py but uses a single, narrow stop pattern:
"IN WITNESS WHEREOF" (case-insensitive).

This is the appropriate choice when the facility agreement reliably ends its
schedule block with an "IN WITNESS WHEREOF" execution clause and you do not
want the broader default stop pattern from parse_schedule.py to fire on
words like "signed" or "executed" that may appear legitimately inside
schedule body text.

Usage:
    python parse_schedule_2.py input.pdf
    python parse_schedule_2.py input.pdf --output schedules.json
    python parse_schedule_2.py input.pdf --show-keys
"""

import argparse
import json
import re
import sys
from pathlib import Path

from parse_sections import extract_blocks
from parse_schedule import parse_schedules


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

_STOP_RE = re.compile(r'^in\s+witness\s+whereof', re.IGNORECASE)


# ─────────────────────────────────────────────
# Public helper
# ─────────────────────────────────────────────

def parse_schedules_v2(blocks: list[str]) -> dict[str, str]:
    """Parse schedules, stopping at 'IN WITNESS WHEREOF'."""
    return parse_schedules(blocks, stop_pattern=_STOP_RE, stop_on_numbered_section=False)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse schedules from a PDF, stopping at 'IN WITNESS WHEREOF'."
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

    blocks    = extract_blocks(str(pdf_path))
    schedules = parse_schedules_v2(blocks)

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
