"""
Unit tests for extract_definitions.py

PDF-based tests use in-memory PDFs built with pymupdf.
Block-level tests use plain string lists to avoid PDF overhead.
"""

import json
import tempfile
from pathlib import Path

import pymupdf
import pytest

from extract_definitions import (
    extract_all_blocks,
    find_definitions_section,
    parse_definitions,
)


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
            y += 50.0
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    doc.save(tmp.name)
    tmp.close()
    return tmp.name


# ─────────────────────────────────────────────
# TestExtractAllBlocks
# ─────────────────────────────────────────────

class TestExtractAllBlocks:
    def test_returns_list_of_strings(self):
        path = _make_pdf([["Some text."]])
        try:
            blocks = extract_all_blocks(path)
            assert isinstance(blocks, list)
            assert all(isinstance(b, str) for b in blocks)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_all_pages_included(self):
        path = _make_pdf([["PAGE ONE"], ["PAGE TWO"], ["PAGE THREE"]])
        try:
            blocks = extract_all_blocks(path)
            full = " ".join(blocks)
            assert "PAGE ONE"   in full
            assert "PAGE TWO"   in full
            assert "PAGE THREE" in full
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_pdf_returns_no_blocks(self):
        doc = pymupdf.Document()
        doc.new_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        tmp.close()
        try:
            assert extract_all_blocks(tmp.name) == []
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# TestFindDefinitionsSection
# ─────────────────────────────────────────────

class TestFindDefinitionsSection:
    def test_basic_heading_found(self):
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Availability Period" means the period...',
            "2. THE FACILITY",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 1
        assert "Availability Period" in result[0]

    def test_heading_excluded_from_result(self):
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Term A" means something.',
        ]
        result = find_definitions_section(blocks)
        assert not any("DEFINITIONS AND INTERPRETATION" in b for b in result)

    def test_stops_at_next_depth1_section(self):
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Business Day" means a day...',
            '"Facility" means the loan...',
            "2. THE FACILITY",
            "Body of facility section.",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 2
        assert "Body of facility section." not in " ".join(result)

    def test_case_insensitive_heading(self):
        for heading in [
            "1. DEFINITIONS AND INTERPRETATION",
            "1. Definitions and Interpretation",
            "1. definitions and interpretations",
            "DEFINITIONS AND INTERPRETATIONS",
            "1.1 Definitions",
            "1.1 DEFINITIONS",
            "Definitions",
        ]:
            blocks = [heading, '"Term" means value.', "2. NEXT SECTION"]
            result = find_definitions_section(blocks)
            assert len(result) >= 1, f"Failed for heading: {heading!r}"

    def test_depth2_heading_found(self):
        blocks = [
            "1. GENERAL",
            "General preamble.",
            "1.1 Definitions",
            '"Business Day" means a working day.',
            "1.2 Interpretation",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 1
        assert "Business Day" in result[0]

    def test_depth2_heading_stops_at_sibling_depth2(self):
        blocks = [
            "1.1 Definitions",
            '"Term A" means value A.',
            '"Term B" means value B.',
            "1.2 Interpretation",
            "Interpretation rules here.",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 2
        assert not any("Interpretation rules" in b for b in result)

    def test_depth2_heading_stops_at_depth1(self):
        blocks = [
            "1.1 Definitions",
            '"Term A" means value A.',
            "2. THE FACILITY",
            "Facility text.",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 1
        assert "Facility text." not in " ".join(result)

    def test_depth1_heading_does_not_stop_at_depth2(self):
        # Existing behaviour preserved: depth-1 definitions heading should
        # NOT be terminated by a depth-2 sibling like "1.2 Interpretation".
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Term A" means something.',
            "1.2 Interpretation",           # depth-2 — should NOT stop
            "Interpretation rules here.",
            "2. THE FACILITY",
        ]
        result = find_definitions_section(blocks)
        assert any("Interpretation rules" in b for b in result)

    def test_section_not_found_returns_empty_list(self):
        blocks = ["1. FACILITY", "Body text.", "2. INTEREST"]
        assert find_definitions_section(blocks) == []

    def test_no_successor_section_returns_all_remaining_blocks(self):
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Term A" means A.',
            '"Term B" means B.',
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 2

    def test_pre_section_blocks_not_included(self):
        blocks = [
            "PREAMBLE TEXT",
            "More preamble.",
            "1. DEFINITIONS AND INTERPRETATION",
            '"Term A" means A.',
            "2. FACILITY",
        ]
        result = find_definitions_section(blocks)
        assert not any("PREAMBLE" in b for b in result)

    def test_multiple_blocks_between_headings_all_included(self):
        blocks = [
            "1. DEFINITIONS AND INTERPRETATION",
            '"Term A" means A.',
            '"Term B" means B.',
            '"Term C" means C.',
            "2. FACILITY",
        ]
        result = find_definitions_section(blocks)
        assert len(result) == 3


# ─────────────────────────────────────────────
# TestParseDefinitions
# ─────────────────────────────────────────────

class TestParseDefinitions:
    def test_single_definition_parsed(self):
        blocks = ['"Availability Period" means the period commencing...']
        result = parse_definitions(blocks)
        assert "Availability Period" in result

    def test_term_is_key_without_quotes(self):
        blocks = ['"Business Day" means a working day.']
        result = parse_definitions(blocks)
        assert "Business Day" in result
        assert '"Business Day"' not in result   # quotes must not be in the key

    def test_full_definition_is_value(self):
        text   = '"Availability Period" means the period commencing on the date.'
        blocks = [text]
        result = parse_definitions(blocks)
        assert result["Availability Period"] == text

    def test_multiple_definitions_all_parsed(self):
        blocks = [
            '"Availability Period" means the period.',
            '"Business Day" means a working day.',
            '"Facility" means the loan facility.',
        ]
        result = parse_definitions(blocks)
        assert set(result.keys()) == {"Availability Period", "Business Day", "Facility"}

    def test_multi_block_definition_folded(self):
        blocks = [
            '"Confidential Information" means any information',
            'received by a party in connection with this Agreement,',
            'whether oral or written.',
            '"Business Day" means a working day.',
        ]
        result = parse_definitions(blocks)
        assert "Confidential Information" in result
        assert "received by a party" in result["Confidential Information"]
        assert "whether oral or written." in result["Confidential Information"]

    def test_multi_block_continuation_not_separate_key(self):
        blocks = [
            '"Confidential Information" means any information',
            'received by a party.',   # continuation
        ]
        result = parse_definitions(blocks)
        assert len(result) == 1
        assert "received by a party." not in result   # not a key

    def test_pre_definition_blocks_discarded(self):
        blocks = [
            "In this Agreement, unless the context otherwise requires:",
            '"Term A" means value A.',
        ]
        result = parse_definitions(blocks)
        assert "Term A" in result
        assert len(result) == 1   # preamble text not parsed as definition

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_definitions([]) == {}

    def test_no_quoted_terms_returns_empty_dict(self):
        blocks = ["Some text.", "More text without quoted terms."]
        assert parse_definitions(blocks) == {}

    def test_smart_quotes_left_right(self):
        # Typographic/curly double quotes: \u201c and \u201d
        blocks = ['\u201cAvailability Period\u201d means the period.']
        result = parse_definitions(blocks)
        assert "Availability Period" in result

    def test_whitespace_normalised_in_value(self):
        # Extra whitespace within a block should be collapsed
        blocks = ['"Term  A"  means   something   here.']
        result = parse_definitions(blocks)
        assert "  " not in result.get("Term  A", result.get("Term A", ""))

    def test_definition_spanning_three_blocks(self):
        blocks = [
            '"Default" means any of the following events:',
            '(a) failure to pay any amount when due;',
            '(b) breach of any representation or warranty.',
            '"Facility" means the loan.',
        ]
        result = parse_definitions(blocks)
        text = result["Default"]
        assert "(a) failure to pay" in text
        assert "(b) breach of" in text
        assert "Facility" not in text   # separate definition, not folded in


# ─────────────────────────────────────────────
# TestIntegration — in-memory PDF
# ─────────────────────────────────────────────

class TestIntegration:
    def _build_pdf(self) -> str:
        pages = [
            [
                "FACILITY AGREEMENT",
                "Dated 1 January 2026",
            ],
            [
                "1. DEFINITIONS AND INTERPRETATION",
                '"Availability Period" means the period commencing on the date of this Agreement.',
                '"Business Day" means a day on which banks are open for business.',
                '"Facility" means the term loan facility made available under this Agreement.',
            ],
            [
                "2. THE FACILITY",
                "The Lender agrees to make available to the Borrower...",
            ],
        ]
        return _make_pdf(pages)

    def test_full_pipeline_extracts_definitions(self):
        path = self._build_pdf()
        try:
            blocks      = extract_all_blocks(path)
            def_blocks  = find_definitions_section(blocks)
            definitions = parse_definitions(def_blocks)
            assert "Availability Period" in definitions
            assert "Business Day"        in definitions
            assert "Facility"            in definitions
        finally:
            Path(path).unlink(missing_ok=True)

    def test_facility_section_body_not_in_definitions(self):
        path = self._build_pdf()
        try:
            blocks      = extract_all_blocks(path)
            def_blocks  = find_definitions_section(blocks)
            definitions = parse_definitions(def_blocks)
            all_text    = " ".join(definitions.values())
            assert "Lender agrees to make available" not in all_text
        finally:
            Path(path).unlink(missing_ok=True)

    def test_preamble_not_in_definitions(self):
        path = self._build_pdf()
        try:
            blocks      = extract_all_blocks(path)
            def_blocks  = find_definitions_section(blocks)
            definitions = parse_definitions(def_blocks)
            assert "FACILITY AGREEMENT" not in definitions
            assert "Dated 1 January 2026" not in definitions
        finally:
            Path(path).unlink(missing_ok=True)

    def test_definition_values_contain_full_text(self):
        path = self._build_pdf()
        try:
            blocks      = extract_all_blocks(path)
            def_blocks  = find_definitions_section(blocks)
            definitions = parse_definitions(def_blocks)
            assert "commencing on the date of this Agreement" in \
                   definitions["Availability Period"]
        finally:
            Path(path).unlink(missing_ok=True)
