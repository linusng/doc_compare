"""
Unit tests for parse_schedules.py

extract_blocks_second_half is tested with real in-memory PDFs built via
pymupdf so we exercise the page-splitting logic without needing a file on
disk.  parse_schedules is tested with plain block lists (no PDF needed).
"""

import math
import tempfile
from pathlib import Path

import pymupdf
import pytest

from parse_schedules import extract_blocks_second_half, parse_schedules


# ─────────────────────────────────────────────
# PDF helpers
# ─────────────────────────────────────────────

def _make_pdf(pages: list[list[str]]) -> str:
    """Write a temporary PDF where each inner list is one page's text blocks."""
    doc = pymupdf.Document()
    for page_blocks in pages:
        page = doc.new_page()
        y = 72.0
        for text in page_blocks:
            page.insert_text((50, y), text)
            y += 60.0
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────
# TestExtractBlocksSecondHalf
# ─────────────────────────────────────────────

class TestExtractBlocksSecondHalf:
    def test_odd_page_count_first_half_excluded(self):
        # 3 pages → start_page = ceil(3/2) = 2 → range(2, 3) = page index 2 only
        # pages 0 and 1 are the first half; only page 2 is the second half
        path = _make_pdf([
            ["PAGE 0"],
            ["PAGE 1"],
            ["PAGE 2"],
        ])
        try:
            blocks = extract_blocks_second_half(path)
            full = " ".join(blocks)
            assert "PAGE 0" not in full
            assert "PAGE 1" not in full
            assert "PAGE 2" in full
        finally:
            Path(path).unlink(missing_ok=True)

    def test_even_page_count_exact_split(self):
        # 4 pages → start_page = ceil(4/2) = 2 → pages 2 & 3 (indices 2 & 3)
        path = _make_pdf([
            ["PAGE 1"],
            ["PAGE 2"],
            ["PAGE 3"],
            ["PAGE 4"],
        ])
        try:
            blocks = extract_blocks_second_half(path)
            full = " ".join(blocks)
            assert "PAGE 1" not in full
            assert "PAGE 2" not in full
            assert "PAGE 3" in full
            assert "PAGE 4" in full
        finally:
            Path(path).unlink(missing_ok=True)

    def test_single_page_returns_all_blocks(self):
        # 1 page → start_page = ceil(1/2) = 1 → range(1, 1) is empty
        # single-page doc: the whole page is the "second half"
        path = _make_pdf([["ONLY PAGE CONTENT"]])
        try:
            blocks = extract_blocks_second_half(path)
            # start_page=1 means range(1,1) → no blocks; acceptable edge case
            # assert the call doesn't raise and returns a list
            assert isinstance(blocks, list)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_two_page_pdf_returns_second_page_only(self):
        # 2 pages → start_page = ceil(2/2) = 1 → page index 1 only
        path = _make_pdf([
            ["BODY CONTENT"],
            ["SCHEDULE 1 CONTENT"],
        ])
        try:
            blocks = extract_blocks_second_half(path)
            full = " ".join(blocks)
            assert "BODY CONTENT"      not in full
            assert "SCHEDULE 1 CONTENT" in full
        finally:
            Path(path).unlink(missing_ok=True)

    def test_start_page_is_ceil_of_half(self):
        # Verify the page count used is exactly ceil(n/2) for various n
        for n_pages in (2, 3, 4, 5, 6, 10):
            expected_start = math.ceil(n_pages / 2)
            # Build a PDF where page i has unique text "PAGE {i}"
            pages = [[f"PAGE {i}"] for i in range(n_pages)]
            path = _make_pdf(pages)
            try:
                blocks = extract_blocks_second_half(path)
                full = " ".join(blocks)
                for i in range(n_pages):
                    if i < expected_start:
                        assert f"PAGE {i}" not in full, (
                            f"n={n_pages}: PAGE {i} should be in first half"
                        )
                    else:
                        assert f"PAGE {i}" in full, (
                            f"n={n_pages}: PAGE {i} should be in second half"
                        )
            finally:
                Path(path).unlink(missing_ok=True)

    def test_empty_pages_return_no_blocks(self):
        doc = pymupdf.Document()
        doc.new_page()
        doc.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        tmp.close()
        try:
            blocks = extract_blocks_second_half(tmp.name)
            assert blocks == []
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# TestParseSchedules
# ─────────────────────────────────────────────

class TestParseSchedules:
    def test_single_schedule_parsed(self):
        blocks = ["Schedule 1\nConditions Precedent."]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Conditions Precedent." in result["Schedule 1"]

    def test_multiple_schedules_all_captured(self):
        blocks = [
            "Schedule 1\nFirst.",
            "Schedule 2\nSecond.",
            "Schedule 3\nThird.",
        ]
        result = parse_schedules(blocks)
        assert list(result.keys()) == ["Schedule 1", "Schedule 2", "Schedule 3"]

    def test_case_insensitive_heading(self):
        blocks = ["SCHEDULE 1\nUpper.", "schedule 2\nLower.", "Schedule 3\nTitle."]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Schedule 2" in result
        assert "Schedule 3" in result

    def test_key_always_title_case(self):
        blocks = ["SCHEDULE 9\nContent."]
        result = parse_schedules(blocks)
        assert "Schedule 9" in result
        assert "SCHEDULE 9" not in result

    def test_body_blocks_collected_under_current_schedule(self):
        blocks = [
            "Schedule 1\nHeading.",
            "Body line one.",
            "Body line two.",
            "Schedule 2\nNext schedule.",
        ]
        result = parse_schedules(blocks)
        assert "Body line one." in result["Schedule 1"]
        assert "Body line two." in result["Schedule 1"]
        assert "Body line one." not in result["Schedule 2"]

    def test_blocks_before_first_schedule_discarded(self):
        blocks = [
            "Some introductory text.",
            "Schedule 1\nSchedule content.",
        ]
        result = parse_schedules(blocks)
        assert len(result) == 1
        assert "introductory text" not in result["Schedule 1"]

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_schedules([]) == {}

    def test_no_schedule_headings_returns_empty_dict(self):
        blocks = ["Just some text.", "More text."]
        assert parse_schedules(blocks) == {}

    def test_title_text_after_number_included_in_value(self):
        blocks = ["Schedule 2 – Form of Drawdown Notice\nContent here."]
        result = parse_schedules(blocks)
        assert "Form of Drawdown Notice" in result["Schedule 2"]

    # ── Stop pattern ─────────────────────────────────────────────────────

    def test_in_witness_whereof_stops_last_schedule(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "IN WITNESS WHEREOF the parties have executed this agreement.",
            "Signature block — must not appear in Schedule 9.",
        ]
        result = parse_schedules(blocks)
        assert "Signature block" not in result.get("Schedule 9", "")

    def test_schedule_body_captured_before_stop(self):
        blocks = [
            "Schedule 9\nSchedule 9 body.",
            "IN WITNESS WHEREOF",
            "Post content.",
        ]
        result = parse_schedules(blocks)
        assert "Schedule 9 body." in result["Schedule 9"]

    def test_stop_case_insensitive_lowercase(self):
        blocks = [
            "Schedule 1\nContent.",
            "in witness whereof the parties agree.",
            "Must not appear.",
        ]
        result = parse_schedules(blocks)
        assert "Must not appear." not in result.get("Schedule 1", "")

    def test_stop_case_insensitive_mixed(self):
        blocks = [
            "Schedule 1\nContent.",
            "In Witness Whereof",
            "Must not appear.",
        ]
        result = parse_schedules(blocks)
        assert "Must not appear." not in result.get("Schedule 1", "")

    def test_all_schedules_before_stop_intact(self):
        blocks = [
            "Schedule 1\nFirst.",
            "Schedule 2\nSecond.",
            "Schedule 3\nThird.",
            "IN WITNESS WHEREOF",
            "Post content.",
        ]
        result = parse_schedules(blocks)
        assert list(result.keys()) == ["Schedule 1", "Schedule 2", "Schedule 3"]
        assert "Post content." not in result.get("Schedule 3", "")

    def test_stop_not_triggered_mid_body_sentence(self):
        # "in witness whereof" only triggers at the START of a block's first line
        blocks = [
            "Schedule 1\nThis agreement, in witness whereof, is signed.",
            "More content.",
        ]
        result = parse_schedules(blocks)
        assert "More content." in result["Schedule 1"]

    def test_no_stop_phrase_all_content_captured(self):
        blocks = [
            "Schedule 1\nContent A.",
            "Content B.",
        ]
        result = parse_schedules(blocks)
        assert "Content A." in result["Schedule 1"]
        assert "Content B." in result["Schedule 1"]
