"""
Unit tests for parse_schedule.py
"""

import re

from parse_schedule import _DEFAULT_STOP_RE, parse_schedules


class TestParseSchedules:
    def test_single_schedule(self):
        blocks = ["Schedule 1\nConditions Precedent text here."]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Conditions Precedent" in result["Schedule 1"]

    def test_multiple_schedules(self):
        blocks = [
            "Schedule 1\nConditions Precedent.",
            "Schedule 2\nForm of Drawdown Notice.",
            "Schedule 3\nForm of Compliance Certificate.",
        ]
        result = parse_schedules(blocks)
        assert list(result.keys()) == ["Schedule 1", "Schedule 2", "Schedule 3"]

    def test_schedule_heading_case_insensitive(self):
        blocks = [
            "SCHEDULE 1\nUpper case heading.",
            "schedule 2\nLower case heading.",
            "Schedule 3\nTitle case heading.",
        ]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Schedule 2" in result
        assert "Schedule 3" in result

    def test_keys_always_title_case(self):
        # Regardless of how the heading appears, the key is "Schedule N"
        blocks = ["SCHEDULE 4\nSome content."]
        result = parse_schedules(blocks)
        assert "Schedule 4" in result
        assert "SCHEDULE 4" not in result

    def test_title_text_after_number_in_value(self):
        blocks = ["Schedule 1 – Conditions Precedent\nList of documents."]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Conditions Precedent" in result["Schedule 1"]

    def test_body_blocks_between_schedules_go_to_current(self):
        blocks = [
            "Schedule 1\nFirst paragraph.",
            "Second paragraph.",
            "Third paragraph.",
            "Schedule 2\nNew schedule.",
        ]
        result = parse_schedules(blocks)
        assert "Second paragraph." in result["Schedule 1"]
        assert "Third paragraph." in result["Schedule 1"]
        assert "Second paragraph." not in result["Schedule 2"]

    def test_blocks_before_first_schedule_discarded(self):
        blocks = [
            "This is the main agreement body.",
            "1. Definitions\nDefined terms here.",
            "Schedule 1\nFirst schedule content.",
        ]
        result = parse_schedules(blocks)
        assert "preamble" not in result
        assert len(result) == 1
        assert "Schedule 1" in result
        assert "main agreement body" not in result["Schedule 1"]

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_schedules([]) == {}

    def test_no_schedules_in_blocks_returns_empty_dict(self):
        blocks = [
            "1. Definitions\nText.",
            "2. Representations\nMore text.",
        ]
        assert parse_schedules(blocks) == {}

    def test_multi_line_schedule_content(self):
        blocks = [
            "Schedule 2\nHeading text.",
            "Paragraph one of schedule two.",
            "Paragraph two of schedule two.",
        ]
        result = parse_schedules(blocks)
        text = result["Schedule 2"]
        assert "Paragraph one" in text
        assert "Paragraph two" in text

    def test_schedule_number_is_preserved(self):
        blocks = [f"Schedule {n}\nContent {n}." for n in range(1, 6)]
        result = parse_schedules(blocks)
        for n in range(1, 6):
            assert f"Schedule {n}" in result
            assert f"Content {n}." in result[f"Schedule {n}"]


# ─────────────────────────────────────────────
# TestStopPatternOptionA
# ─────────────────────────────────────────────

class TestStopPatternOptionA:
    """Option A: stop_pattern terminates the last schedule."""

    def _blocks_with_trailer(self, trailer_line: str) -> list[str]:
        return [
            "Schedule 8\nSchedule 8 body.",
            "Schedule 9\nSchedule 9 body.",
            trailer_line,
            "Post-schedule content that must NOT appear in Schedule 9.",
        ]

    def test_signed_by_stops_collection(self):
        blocks = self._blocks_with_trailer("SIGNED BY the parties as follows:")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_executed_stops_collection(self):
        blocks = self._blocks_with_trailer("Executed as a deed by ABC Bank")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_in_witness_whereof_stops_collection(self):
        blocks = self._blocks_with_trailer("IN WITNESS WHEREOF the parties have executed")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_signatories_stops_collection(self):
        blocks = self._blocks_with_trailer("Signatories")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_authorised_signatory_stops_collection(self):
        blocks = self._blocks_with_trailer("Authorised Signatory:")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_annexure_stops_collection(self):
        blocks = self._blocks_with_trailer("Annexure A – Additional Terms")
        result = parse_schedules(blocks)
        assert "Post-schedule content" not in result.get("Schedule 9", "")

    def test_schedule_9_body_still_captured_before_stop(self):
        blocks = self._blocks_with_trailer("SIGNED BY the parties as follows:")
        result = parse_schedules(blocks)
        assert "Schedule 9 body." in result["Schedule 9"]

    def test_stop_pattern_none_disables_option_a(self):
        # With stop_pattern=None the trailer is absorbed into Schedule 9
        blocks = self._blocks_with_trailer("SIGNED BY the parties:")
        result = parse_schedules(blocks, stop_pattern=None)
        assert "Post-schedule content" in result.get("Schedule 9", "")

    def test_custom_stop_pattern(self):
        blocks = [
            "Schedule 1\nBody of schedule 1.",
            "END OF SCHEDULES",
            "This must not appear in Schedule 1.",
        ]
        result = parse_schedules(
            blocks,
            stop_pattern=re.compile(r'^END OF SCHEDULES', re.IGNORECASE),
        )
        assert "This must not appear" not in result["Schedule 1"]
        assert "END OF SCHEDULES" not in result["Schedule 1"]

    def test_stop_before_any_schedule_is_harmless(self):
        # Stop pattern fires before any schedule — since we are not inside a
        # schedule yet, it is ignored and parsing continues normally.
        blocks = [
            "SIGNED BY the parties:",
            "Schedule 1\nSome content.",
        ]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Some content." in result["Schedule 1"]

    def test_default_stop_re_does_not_match_mid_body_text(self):
        # Words like "signed" appearing mid-sentence inside body should not
        # trigger the stop (the regex anchors at the start of the line).
        blocks = [
            "Schedule 1\nThis agreement was signed last year.",
            "More body text.",
        ]
        result = parse_schedules(blocks)
        assert "More body text." in result["Schedule 1"]


# ─────────────────────────────────────────────
# TestStopOnNumberedSectionOptionB
# ─────────────────────────────────────────────

class TestStopOnNumberedSectionOptionB:
    """Option B: stop_on_numbered_section terminates when a depth-1 section appears."""

    def test_numbered_section_after_last_schedule_stops_collection(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "1. Execution\nExecution block text.",
            "Signature lines here.",
        ]
        result = parse_schedules(blocks, stop_on_numbered_section=True)
        assert "Execution block text." not in result.get("Schedule 9", "")
        assert "Signature lines" not in result.get("Schedule 9", "")

    def test_schedule_9_body_captured_before_section_trigger(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "1. Execution\nExecution block.",
        ]
        result = parse_schedules(blocks, stop_on_numbered_section=True)
        assert "Schedule 9 body." in result["Schedule 9"]

    def test_option_b_off_by_default_numbered_section_absorbed(self):
        # Without the flag, a numbered section is just body text
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "1. Execution\nThis would be absorbed.",
        ]
        result = parse_schedules(blocks)  # stop_on_numbered_section defaults to False
        assert "This would be absorbed." in result["Schedule 9"]

    def test_depth_2_section_does_not_trigger_option_b(self):
        # Only depth-1 headings (e.g. "1.") trigger; "1.1" should not
        blocks = [
            "Schedule 1\nSchedule content.",
            "1.1 Sub-condition\nDetail text.",
            "More schedule content.",
        ]
        result = parse_schedules(blocks, stop_on_numbered_section=True)
        assert "More schedule content." in result["Schedule 1"]

    def test_option_a_and_b_both_active(self):
        # When both are enabled, whichever fires first wins
        blocks = [
            "Schedule 9\nBody.",
            "SIGNED BY ABC Bank",   # Option A fires here
            "1. Execution",         # Option B would fire here but never reached
            "Post content.",
        ]
        result = parse_schedules(blocks, stop_on_numbered_section=True)
        assert "Post content." not in result.get("Schedule 9", "")
        assert "Body." in result["Schedule 9"]
