"""
Unit tests for parse_sections_2.py
"""

from pathlib import Path

import pytest

from parse_sections_2 import _build_heading_key, parse_sections

MOCK_PDF = Path(__file__).parent.parent / "files" / "mock_document.pdf"


# ─────────────────────────────────────────────
# TestBuildHeadingKey
# ─────────────────────────────────────────────

class TestBuildHeadingKey:
    def test_top_level_with_trailing_dot(self):
        assert _build_heading_key("1", "1. DEFINITIONS AND INTERPRETATION") == \
               "1 DEFINITIONS AND INTERPRETATION"

    def test_subsection_no_trailing_dot(self):
        assert _build_heading_key("1.1", "1.1 Definitions") == "1.1 Definitions"

    def test_deep_section_number_stripped(self):
        assert _build_heading_key("3.3", "3.3 Financial Covenants") == \
               "3.3 Financial Covenants"

    def test_no_heading_text_falls_back_to_number(self):
        # First line is only the section number with nothing after it
        assert _build_heading_key("2", "2.") == "2"

    def test_preserves_capitalisation(self):
        key = _build_heading_key("6.1", "6.1 Non-Payment")
        assert key == "6.1 Non-Payment"


# ─────────────────────────────────────────────
# TestParseSections
# ─────────────────────────────────────────────

class TestParseSections:
    def test_keys_include_heading_text(self):
        blocks = ["1.1 Definitions\nSome definitions here."]
        result = parse_sections(blocks)
        assert "1.1 Definitions" in result

    def test_numeric_only_key_absent(self):
        # "1.1" alone should NOT be a key — heading key replaces it
        blocks = ["1.1 Definitions\nSome definitions here."]
        result = parse_sections(blocks)
        assert "1.1" not in result

    def test_top_level_trailing_dot_normalised(self):
        blocks = ["1. Introduction\nText here."]
        result = parse_sections(blocks)
        assert "1 Introduction" in result
        assert "1." not in result

    def test_preamble_key_unchanged(self):
        blocks = ["This is preamble text.", "1. Section\nContent."]
        result = parse_sections(blocks)
        assert "preamble" in result

    def test_deep_sections_folded_into_parent(self):
        blocks = [
            "3.3 Financial Covenants\nGeneral text.",
            "3.3.1 DSCR\nDSCR shall be 1.25x.",
            "3.3.2 LTV\nLTV shall be 65%.",
        ]
        result = parse_sections(blocks)
        assert "3.3 Financial Covenants" in result
        assert not any("3.3.1" in k for k in result)
        assert not any("3.3.2" in k for k in result)

    def test_deep_section_text_in_parent_value(self):
        blocks = [
            "3.3 Financial Covenants\nGeneral text.",
            "3.3.1 DSCR\nDSCR shall not fall below 1.25x.",
        ]
        result = parse_sections(blocks)
        assert "1.25x" in result["3.3 Financial Covenants"]

    def test_body_text_between_headings_goes_to_current(self):
        blocks = [
            "2. The Facility\nSubject to the terms:",
            "All conditions shall be met.",
            "2.1 Amount\nSGD 50 million.",
        ]
        result = parse_sections(blocks)
        assert "All conditions shall be met." in result["2 The Facility"]

    def test_multiple_sections_all_have_heading_keys(self):
        blocks = [
            "1. Introduction\nText A.",
            "2. Methods\nText B.",
            "3. Results\nText C.",
        ]
        result = parse_sections(blocks)
        assert set(result.keys()) == {"1 Introduction", "2 Methods", "3 Results"}

    def test_empty_blocks_returns_empty_dict(self):
        assert parse_sections([]) == {}

    def test_section_heading_present_in_value(self):
        # The full first line (including the number) should appear in the value text
        blocks = ["4.2 Voluntary Prepayment\nBorrower may prepay."]
        result = parse_sections(blocks)
        assert "4.2 Voluntary Prepayment" in result["4.2 Voluntary Prepayment"]


# ─────────────────────────────────────────────
# TestIntegration
# ─────────────────────────────────────────────

@pytest.mark.skipif(
    not MOCK_PDF.exists(),
    reason="mock_document.pdf not found -- run generate_mock_pdf.py first",
)
class TestIntegration:
    @pytest.fixture(scope="class")
    def sections(self):
        from parse_sections import extract_blocks
        from parse_sections_2 import parse_sections
        return parse_sections(extract_blocks(str(MOCK_PDF)))

    def test_all_top_level_heading_keys_present(self, sections):
        keys = set(sections.keys())
        for fragment in ["DEFINITIONS", "FACILITY", "INTEREST", "REPAYMENT",
                         "REPRESENTATIONS", "DEFAULT"]:
            assert any(fragment in k for k in keys), \
                f"No key containing '{fragment}'"

    def test_subsection_heading_keys_present(self, sections):
        keys = set(sections.keys())
        for fragment in ["Definitions", "Interest Rate", "Financial Covenants",
                         "Repayment Schedule", "Non-Payment"]:
            assert any(fragment in k for k in keys), \
                f"No key containing '{fragment}'"

    def test_no_bare_numeric_keys(self, sections):
        for key in sections:
            if key == "preamble":
                continue
            # Every non-preamble key must have text beyond just the number
            parts = key.split(" ", 1)
            assert len(parts) == 2 and parts[1].strip(), \
                f"Key '{key}' has no heading text"

    def test_deep_sections_not_separate_keys(self, sections):
        for key in sections:
            assert "3.3.1" not in key
            assert "3.3.2" not in key
            assert "3.3.3" not in key

    def test_dscr_content_in_financial_covenants_key(self, sections):
        cov_key = next(k for k in sections if "Financial Covenants" in k)
        assert "DSCR" in sections[cov_key]
        assert "LTV"  in sections[cov_key]
