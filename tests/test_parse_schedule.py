"""
Unit tests for parse_schedule.py
"""

from parse_schedule import parse_schedules


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
