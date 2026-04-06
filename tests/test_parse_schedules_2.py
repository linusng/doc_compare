"""
Unit tests for parse_schedules_2.py

extract_blocks_second_half is tested with real in-memory PDFs (pymupdf).
parse_schedules is tested with plain (text, page_number) lists.
"""

import math
import tempfile
from pathlib import Path

import pymupdf
import pytest

from parse_schedules_2 import extract_blocks_second_half, parse_schedules


# ─────────────────────────────────────────────
# PDF helper
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
    def test_returns_list_of_tuples(self):
        path = _make_pdf([["Page one content."], ["Page two content."]])
        try:
            blocks = extract_blocks_second_half(path)
            assert isinstance(blocks, list)
            for item in blocks:
                assert isinstance(item, tuple)
                assert len(item) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_second_element_is_page_number(self):
        path = _make_pdf([["First."], ["Second."]])
        try:
            blocks = extract_blocks_second_half(path)
            for _, page_number in blocks:
                assert isinstance(page_number, int)
                assert page_number >= 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_page_numbers_are_one_based(self):
        # 2-page PDF → start_page = ceil(2/2) = 1 → only page index 1 (page 2)
        path = _make_pdf([["First page."], ["Second page."]])
        try:
            blocks = extract_blocks_second_half(path)
            page_numbers = {pn for _, pn in blocks}
            assert 2 in page_numbers   # page index 1 → page number 2
            assert 1 not in page_numbers
        finally:
            Path(path).unlink(missing_ok=True)

    def test_first_half_excluded(self):
        path = _make_pdf([["FIRST HALF"], ["SECOND HALF"]])
        try:
            blocks = extract_blocks_second_half(path)
            texts = [t for t, _ in blocks]
            assert not any("FIRST HALF" in t for t in texts)
            assert any("SECOND HALF" in t for t in texts)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_multi_page_second_half_correct_page_numbers(self):
        # 4 pages → start_page = 2 → pages 3 and 4 (1-based)
        path = _make_pdf([
            ["PAGE ONE"],
            ["PAGE TWO"],
            ["PAGE THREE"],
            ["PAGE FOUR"],
        ])
        try:
            blocks = extract_blocks_second_half(path)
            page_numbers = sorted({pn for _, pn in blocks})
            assert 1 not in page_numbers
            assert 2 not in page_numbers
            assert 3 in page_numbers
            assert 4 in page_numbers
        finally:
            Path(path).unlink(missing_ok=True)

    def test_page_number_matches_content(self):
        # Verify that each block is tagged with the correct page number
        path = _make_pdf([
            ["SKIP THIS"],           # page 1 — first half
            ["SCHEDULE 1 CONTENT"],  # page 2 — second half
        ])
        try:
            blocks = extract_blocks_second_half(path)
            for text, page_number in blocks:
                if "SCHEDULE 1 CONTENT" in text:
                    assert page_number == 2
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

    def test_start_page_is_ceil_of_half(self):
        for n_pages in (2, 3, 4, 5, 6, 10):
            expected_start = math.ceil(n_pages / 2)
            pages = [[f"PAGE {i + 1}"] for i in range(n_pages)]
            path = _make_pdf(pages)
            try:
                blocks = extract_blocks_second_half(path)
                seen_pages = {pn for _, pn in blocks}
                for i in range(n_pages):
                    page_num = i + 1
                    if i < expected_start:
                        assert page_num not in seen_pages, (
                            f"n={n_pages}: page {page_num} should be first half"
                        )
                    else:
                        assert page_num in seen_pages, (
                            f"n={n_pages}: page {page_num} should be second half"
                        )
            finally:
                Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# TestParseSchedules
# ─────────────────────────────────────────────

class TestParseSchedules:
    # ── Output structure ─────────────────────────────────────────────────

    def test_returns_dict_of_dicts(self):
        blocks = [("Schedule 1\nContent.", 5)]
        result = parse_schedules(blocks)
        assert isinstance(result, dict)
        for value in result.values():
            assert isinstance(value, dict)
            assert "text" in value
            assert "page_numbers" in value

    def test_text_value_is_string(self):
        blocks = [("Schedule 1\nContent.", 5)]
        result = parse_schedules(blocks)
        assert isinstance(result["Schedule 1"]["text"], str)

    def test_page_numbers_value_is_list_of_ints(self):
        blocks = [("Schedule 1\nContent.", 5)]
        result = parse_schedules(blocks)
        pns = result["Schedule 1"]["page_numbers"]
        assert isinstance(pns, list)
        assert all(isinstance(p, int) for p in pns)

    # ── Content correctness ──────────────────────────────────────────────

    def test_single_schedule_text_captured(self):
        blocks = [("Schedule 1\nConditions Precedent.", 10)]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Conditions Precedent." in result["Schedule 1"]["text"]

    def test_multiple_schedules_all_keys_present(self):
        blocks = [
            ("Schedule 1\nFirst.", 10),
            ("Schedule 2\nSecond.", 12),
            ("Schedule 3\nThird.", 14),
        ]
        result = parse_schedules(blocks)
        assert list(result.keys()) == ["Schedule 1", "Schedule 2", "Schedule 3"]

    def test_body_blocks_folded_into_current_schedule(self):
        blocks = [
            ("Schedule 1\nHeading.", 5),
            ("Body line one.", 5),
            ("Body line two.", 6),
            ("Schedule 2\nNext.", 7),
        ]
        result = parse_schedules(blocks)
        assert "Body line one." in result["Schedule 1"]["text"]
        assert "Body line two." in result["Schedule 1"]["text"]
        assert "Body line one." not in result["Schedule 2"]["text"]

    def test_blocks_before_first_schedule_discarded(self):
        blocks = [
            ("Introductory text.", 4),
            ("Schedule 1\nContent.", 5),
        ]
        result = parse_schedules(blocks)
        assert len(result) == 1
        assert "Introductory text." not in result["Schedule 1"]["text"]

    def test_case_insensitive_schedule_heading(self):
        blocks = [
            ("SCHEDULE 1\nUpper.", 5),
            ("schedule 2\nLower.", 6),
        ]
        result = parse_schedules(blocks)
        assert "Schedule 1" in result
        assert "Schedule 2" in result

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_schedules([]) == {}

    def test_no_schedule_headings_returns_empty_dict(self):
        blocks = [("Just some text.", 5), ("More text.", 6)]
        assert parse_schedules(blocks) == {}

    # ── Page numbers ─────────────────────────────────────────────────────

    def test_page_number_recorded_for_single_block(self):
        blocks = [("Schedule 1\nContent.", 14)]
        result = parse_schedules(blocks)
        assert result["Schedule 1"]["page_numbers"] == [14]

    def test_page_numbers_span_multiple_pages(self):
        blocks = [
            ("Schedule 1\nFirst block.", 14),
            ("Second block.",            15),
            ("Third block.",             16),
        ]
        result = parse_schedules(blocks)
        assert result["Schedule 1"]["page_numbers"] == [14, 15, 16]

    def test_page_numbers_are_unique_and_sorted(self):
        blocks = [
            ("Schedule 1\nBlock A.", 14),
            ("Block B.",             14),   # same page as heading
            ("Block C.",             16),
            ("Block D.",             15),   # out of order
        ]
        result = parse_schedules(blocks)
        assert result["Schedule 1"]["page_numbers"] == [14, 15, 16]

    def test_page_numbers_are_one_based(self):
        blocks = [("Schedule 1\nContent.", 1)]
        result = parse_schedules(blocks)
        assert 1 in result["Schedule 1"]["page_numbers"]

    def test_different_schedules_have_different_page_numbers(self):
        blocks = [
            ("Schedule 1\nContent.", 5),
            ("Schedule 2\nContent.", 9),
        ]
        result = parse_schedules(blocks)
        assert result["Schedule 1"]["page_numbers"] == [5]
        assert result["Schedule 2"]["page_numbers"] == [9]

    def test_page_numbers_not_shared_across_schedules(self):
        blocks = [
            ("Schedule 1\nBlock.", 5),
            ("Schedule 2\nBlock.", 6),
        ]
        result = parse_schedules(blocks)
        assert 5 not in result["Schedule 2"]["page_numbers"]
        assert 6 not in result["Schedule 1"]["page_numbers"]

    # ── Stop pattern ─────────────────────────────────────────────────────

    def test_in_witness_whereof_stops_last_schedule(self):
        blocks = [
            ("Schedule 9\nBody.",                                  20),
            ("IN WITNESS WHEREOF the parties have executed this.", 21),
            ("Signature block.",                                   21),
        ]
        result = parse_schedules(blocks)
        assert "Signature block." not in result.get("Schedule 9", {}).get("text", "")

    def test_stop_does_not_include_stop_page_in_page_numbers(self):
        blocks = [
            ("Schedule 9\nBody.", 20),
            ("IN WITNESS WHEREOF", 21),
            ("Signature block.",   21),
        ]
        result = parse_schedules(blocks)
        assert 21 not in result["Schedule 9"]["page_numbers"]

    def test_schedule_body_captured_before_stop(self):
        blocks = [
            ("Schedule 9\nBody.", 20),
            ("IN WITNESS WHEREOF", 21),
        ]
        result = parse_schedules(blocks)
        assert "Body." in result["Schedule 9"]["text"]

    def test_stop_case_insensitive(self):
        blocks = [
            ("Schedule 1\nContent.", 5),
            ("in witness whereof",   6),
            ("Must not appear.",     6),
        ]
        result = parse_schedules(blocks)
        assert "Must not appear." not in result.get("Schedule 1", {}).get("text", "")
