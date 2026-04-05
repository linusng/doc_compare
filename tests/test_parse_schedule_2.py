"""
Unit tests for parse_schedule_2.py
"""

from parse_schedule_2 import parse_schedules_v2


class TestParseSchedulesV2:
    def test_in_witness_whereof_stops_last_schedule(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "IN WITNESS WHEREOF the parties have executed this agreement.",
            "Signature block content.",
        ]
        result = parse_schedules_v2(blocks)
        assert "Signature block content." not in result.get("Schedule 9", "")

    def test_schedule_9_body_captured_before_stop(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "IN WITNESS WHEREOF the parties have executed this agreement.",
            "Signature block content.",
        ]
        result = parse_schedules_v2(blocks)
        assert "Schedule 9 body." in result["Schedule 9"]

    def test_case_insensitive_lowercase(self):
        blocks = [
            "Schedule 1\nContent.",
            "in witness whereof the parties agree.",
            "Must not appear.",
        ]
        result = parse_schedules_v2(blocks)
        assert "Must not appear." not in result.get("Schedule 1", "")

    def test_case_insensitive_mixed(self):
        blocks = [
            "Schedule 1\nContent.",
            "In Witness Whereof the parties agree.",
            "Must not appear.",
        ]
        result = parse_schedules_v2(blocks)
        assert "Must not appear." not in result.get("Schedule 1", "")

    def test_all_schedules_before_stop_are_captured(self):
        blocks = [
            "Schedule 1\nFirst schedule.",
            "Schedule 2\nSecond schedule.",
            "Schedule 3\nThird schedule.",
            "IN WITNESS WHEREOF",
            "Post-schedule content.",
        ]
        result = parse_schedules_v2(blocks)
        assert list(result.keys()) == ["Schedule 1", "Schedule 2", "Schedule 3"]
        assert "Post-schedule content." not in result.get("Schedule 3", "")

    def test_signed_mid_text_does_not_stop(self):
        # "signed" or "executed" alone must NOT stop — only "in witness whereof" does
        blocks = [
            "Schedule 1\nThis form must be signed by the borrower.",
            "More schedule content.",
        ]
        result = parse_schedules_v2(blocks)
        assert "More schedule content." in result["Schedule 1"]

    def test_executed_mid_text_does_not_stop(self):
        blocks = [
            "Schedule 1\nThe deed was executed on the date hereof.",
            "More schedule content.",
        ]
        result = parse_schedules_v2(blocks)
        assert "More schedule content." in result["Schedule 1"]

    def test_no_stop_phrase_all_content_captured(self):
        blocks = [
            "Schedule 1\nContent A.",
            "Content B.",
        ]
        result = parse_schedules_v2(blocks)
        assert "Content A." in result["Schedule 1"]
        assert "Content B." in result["Schedule 1"]

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_schedules_v2([]) == {}
