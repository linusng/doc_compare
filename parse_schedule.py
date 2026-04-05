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

Termination
-----------
Two complementary mechanisms prevent the last schedule from absorbing
content that belongs to the execution / signature section:

Option A — stop_pattern (primary, on by default)
    When a block's first line matches the stop pattern, the current schedule
    is flushed and collection halts.  The default covers the execution and
    signature blocks that appear at the end of most facility agreements
    ("SIGNED BY", "EXECUTED by", "IN WITNESS WHEREOF", "Signatories", etc.).
    Override with a custom compiled regex when your document uses different
    wording, or pass ``stop_pattern=None`` to disable.

Option B — stop_on_numbered_section (secondary, off by default)
    When enabled, a depth-1 numbered section heading (e.g. "1.", "15.") that
    appears after the first schedule is treated as a terminator.  Useful when
    the document reverts to numbered body sections after the schedule block.
    Disabled by default because numbered paragraphs *inside* schedule text
    (e.g. "1. The following conditions…") use the same pattern and would
    cause false positives.

Usage:
    python parse_schedule.py input.pdf
    python parse_schedule.py input.pdf --output schedules.json
    python parse_schedule.py input.pdf --show-keys
    python parse_schedule.py input.pdf --stop-pattern "^(?:SIGNED|Execution)"
    python parse_schedule.py input.pdf --stop-on-numbered-section
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Re-use the shared PDF extraction utility — no duplication
from parse_sections import extract_blocks, _SECTION_RE, _section_depth


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

# Matches a schedule heading at the very start of a line, e.g.:
#   "Schedule 1"  "SCHEDULE 1"  "Schedule 1 – Conditions Precedent"
# Group 1 → schedule number (digits only)
# Group 2 → optional title text on the same line (may be empty)
_SCHEDULE_RE = re.compile(r'^schedule\s+(\d+)\b\s*(.*)', re.IGNORECASE)

# Option A — default stop pattern.
# Matches the opening line of execution / signature blocks common in
# facility agreements.  Case-insensitive.
_DEFAULT_STOP_RE = re.compile(
    r'^(?:'
    r'signed'           # "SIGNED BY", "Signed by"
    r'|executed'        # "EXECUTED by", "Executed as a deed"
    r'|in\s+witness'    # "IN WITNESS WHEREOF"
    r'|signator'        # "Signatories", "SIGNATORY"
    r'|authoris'        # "Authorised Signatory" (British spelling)
    r'|authoriz'        # "Authorized Signatory" (American spelling)
    r'|annex'           # "Annexure", "ANNEX"
    r')',
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# Parsing
# ─────────────────────────────────────────────

def parse_schedules(
    blocks: list[str],
    stop_pattern: re.Pattern | None = _DEFAULT_STOP_RE,
    stop_on_numbered_section: bool = False,
) -> dict[str, str]:
    """
    Walk blocks in order and build a Schedule-key → text mapping.

    Parameters
    ----------
    blocks:
        Text blocks from the PDF (as returned by ``extract_blocks``).
    stop_pattern:
        Option A.  A compiled regex; when a block's first line matches it,
        the current schedule is flushed and parsing stops.  Pass ``None``
        to disable.  Defaults to ``_DEFAULT_STOP_RE``.
    stop_on_numbered_section:
        Option B.  When ``True``, a depth-1 numbered section heading
        ("1.", "2.", …) appearing after the first schedule is also treated
        as a terminator.  Defaults to ``False`` to avoid false positives
        from numbered paragraphs inside schedule text.

    Rules
    -----
    - A block whose first line matches _SCHEDULE_RE starts a new schedule.
    - All subsequent blocks belong to that schedule until:
        (a) the next schedule heading appears, OR
        (b) a stop_pattern match is found (Option A), OR
        (c) a depth-1 section number is found and stop_on_numbered_section
            is True (Option B).
    - Blocks before the first schedule heading are discarded.
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

        # ── Option A: stop pattern ────────────────────────────────────────
        if stop_pattern and current_key is not None:
            if stop_pattern.match(first_line):
                _flush()
                break

        # ── Option B: numbered section reversion ─────────────────────────
        if stop_on_numbered_section and current_key is not None:
            sec_match = _SECTION_RE.match(first_line)
            if sec_match and _section_depth(sec_match.group(1).rstrip(".")) == 1:
                _flush()
                break

        # ── Schedule heading ──────────────────────────────────────────────
        sched_match = _SCHEDULE_RE.match(first_line)
        if sched_match:
            _flush()
            number = sched_match.group(1)
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
    parser.add_argument(
        "--stop-pattern",
        default=None,
        help=(
            "Regex that, when matched at the start of a block's first line, "
            "halts schedule collection (Option A). "
            "Replaces the built-in default; pass an empty string to disable."
        ),
    )
    parser.add_argument(
        "--stop-on-numbered-section",
        action="store_true",
        default=False,
        help=(
            "Halt collection when a depth-1 numbered section heading "
            "appears after the first schedule (Option B). "
            "Off by default — enable only when schedules contain no "
            "numbered paragraphs."
        ),
    )
    args = parser.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"Error: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve --stop-pattern: explicit arg overrides the default
    if args.stop_pattern is not None:
        stop_pattern = re.compile(args.stop_pattern, re.IGNORECASE) if args.stop_pattern else None
    else:
        stop_pattern = _DEFAULT_STOP_RE

    blocks    = extract_blocks(str(pdf_path))
    schedules = parse_schedules(
        blocks,
        stop_pattern=stop_pattern,
        stop_on_numbered_section=args.stop_on_numbered_section,
    )

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
