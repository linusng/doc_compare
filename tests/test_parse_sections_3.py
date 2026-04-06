"""
Unit tests for parse_sections_3.py

extract_blocks is tested with real in-memory PDFs (pymupdf).
parse_sections is tested with plain (text, page_number) lists.
"""

import tempfile
from pathlib import Path

import pymupdf
import pytest

from parse_sections_3 import extract_blocks, parse_sections


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
# TestExtractBlocks
# ─────────────────────────────────────────────

class TestExtractBlocks:
    def test_returns_list_of_tuples(self):
        path = _make_pdf([["Some text."]])
        try:
            blocks = extract_blocks(path)
            assert isinstance(blocks, list)
            for item in blocks:
                assert isinstance(item, tuple) and len(item) == 2
        finally:
            Path(path).unlink(missing_ok=True)

    def test_second_element_is_one_based_page_number(self):
        path = _make_pdf([["Page one."], ["Page two."]])
        try:
            blocks = extract_blocks(path)
            page_numbers = {pn for _, pn in blocks}
            assert 1 in page_numbers
            assert 2 in page_numbers
            assert 0 not in page_numbers
        finally:
            Path(path).unlink(missing_ok=True)

    def test_all_pages_included(self):
        # Unlike parse_schedules_3, the whole document is extracted
        path = _make_pdf([["FIRST PAGE"], ["SECOND PAGE"], ["THIRD PAGE"]])
        try:
            blocks = extract_blocks(path)
            full = " ".join(t for t, _ in blocks)
            assert "FIRST PAGE"  in full
            assert "SECOND PAGE" in full
            assert "THIRD PAGE"  in full
        finally:
            Path(path).unlink(missing_ok=True)

    def test_page_number_matches_content(self):
        path = _make_pdf([["ALPHA"], ["BETA"], ["GAMMA"]])
        try:
            blocks = extract_blocks(path)
            for text, pn in blocks:
                if "ALPHA" in text:
                    assert pn == 1
                if "BETA" in text:
                    assert pn == 2
                if "GAMMA" in text:
                    assert pn == 3
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_pdf_returns_no_blocks(self):
        doc = pymupdf.Document()
        doc.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        tmp.close()
        try:
            assert extract_blocks(tmp.name) == []
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# TestParseSections — output structure
# ─────────────────────────────────────────────

class TestParseSectionsStructure:
    def test_returns_dict_of_dicts(self):
        blocks = [("1. Introduction\nText.", 1)]
        result = parse_sections(blocks)
        for value in result.values():
            assert isinstance(value, dict)
            assert "text" in value
            assert "page_numbers" in value

    def test_text_is_string(self):
        blocks = [("1. Introduction\nText.", 1)]
        result = parse_sections(blocks)
        assert isinstance(result["1"]["text"], str)

    def test_page_numbers_is_list_of_ints(self):
        blocks = [("1. Introduction\nText.", 1)]
        result = parse_sections(blocks)
        pns = result["1"]["page_numbers"]
        assert isinstance(pns, list)
        assert all(isinstance(p, int) for p in pns)


# ─────────────────────────────────────────────
# TestParseSections — section keys
# ─────────────────────────────────────────────

class TestParseSectionsKeys:
    def test_top_level_section_key(self):
        blocks = [("1. Introduction\nText.", 1)]
        result = parse_sections(blocks)
        assert "1" in result
        assert "1." not in result

    def test_subsection_key(self):
        blocks = [
            ("1. Introduction\nText.", 1),
            ("1.1 Background\nDetail.", 1),
        ]
        result = parse_sections(blocks)
        assert "1" in result
        assert "1.1" in result

    def test_deep_section_folded_into_parent(self):
        blocks = [
            ("1.1 Payment\nGeneral.", 2),
            ("1.1.1 Principal\nPrincipal text.", 2),
        ]
        result = parse_sections(blocks)
        assert "1.1" in result
        assert "1.1.1" not in result

    def test_deep_section_text_in_parent(self):
        blocks = [
            ("3.3 Covenants\nGeneral.", 5),
            ("3.3.1 DSCR\nDSCR floor is 1.25x.", 5),
        ]
        result = parse_sections(blocks)
        assert "DSCR floor is 1.25x." in result["3.3"]["text"]

    def test_preamble_key_for_pre_section_blocks(self):
        blocks = [
            ("This agreement is dated 1 January 2026.", 1),
            ("1. Definitions\nDefined terms.", 2),
        ]
        result = parse_sections(blocks)
        assert "preamble" in result
        assert "1 January 2026" in result["preamble"]["text"]

    def test_bare_integer_not_a_key(self):
        blocks = [
            ("1. Introduction\nText.", 1),
            ("42", 2),
            ("More text.", 2),
        ]
        result = parse_sections(blocks)
        assert "42" not in result
        assert "More text." in result["1"]["text"]

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_sections([]) == {}


# ─────────────────────────────────────────────
# TestParseSections — page numbers
# ─────────────────────────────────────────────

class TestParseSectionsPageNumbers:
    def test_single_block_single_page(self):
        blocks = [("1. Introduction\nText.", 3)]
        result = parse_sections(blocks)
        assert result["1"]["page_numbers"] == [3]

    def test_multi_block_spanning_pages(self):
        blocks = [
            ("1. Introduction\nFirst block.", 3),
            ("Continued text.", 4),
            ("More text.", 5),
        ]
        result = parse_sections(blocks)
        assert result["1"]["page_numbers"] == [3, 4, 5]

    def test_page_numbers_unique_and_sorted(self):
        blocks = [
            ("2. Facility\nBlock A.", 7),
            ("Block B.", 7),    # same page
            ("Block C.", 9),
            ("Block D.", 8),    # out of order
        ]
        result = parse_sections(blocks)
        assert result["2"]["page_numbers"] == [7, 8, 9]

    def test_page_numbers_are_one_based(self):
        blocks = [("1. Introduction\nText.", 1)]
        result = parse_sections(blocks)
        assert 1 in result["1"]["page_numbers"]

    def test_different_sections_have_independent_page_numbers(self):
        blocks = [
            ("1. Introduction\nText.", 1),
            ("2. Facility\nText.",     4),
        ]
        result = parse_sections(blocks)
        assert result["1"]["page_numbers"] == [1]
        assert result["2"]["page_numbers"] == [4]
        assert 4 not in result["1"]["page_numbers"]
        assert 1 not in result["2"]["page_numbers"]

    def test_preamble_page_numbers_recorded(self):
        blocks = [
            ("Preamble text.", 1),
            ("1. Definitions\nText.", 2),
        ]
        result = parse_sections(blocks)
        assert result["preamble"]["page_numbers"] == [1]

    def test_deep_section_pages_folded_into_parent(self):
        blocks = [
            ("3.3 Covenants\nGeneral.", 10),
            ("3.3.1 DSCR\nDetail.",     11),
            ("3.3.2 LTV\nDetail.",      12),
        ]
        result = parse_sections(blocks)
        assert result["3.3"]["page_numbers"] == [10, 11, 12]

    def test_body_text_pages_included_in_section(self):
        blocks = [
            ("1. Introduction\nHeading.", 1),
            ("Body on page 2.",           2),
            ("Body on page 3.",           3),
        ]
        result = parse_sections(blocks)
        assert 2 in result["1"]["page_numbers"]
        assert 3 in result["1"]["page_numbers"]
