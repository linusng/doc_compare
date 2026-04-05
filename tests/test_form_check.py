"""
Unit tests for form_check.py

Integration tests require both mock PDFs:
    python generate_mock_proposal.py   -> files/mock_proposal.pdf
    python generate_mock_pdf.py        -> files/mock_document.pdf
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pymupdf
import pytest
from fpdf import FPDF, XPos, YPos
from langchain_core.documents import Document

from form_check import (
    SCORE_THRESHOLD,
    CheckResult,
    FieldEntry,
    FieldMatch,
    _extract_heading,
    _is_section_label,
    _match_label,
    check_field,
    extract_form_fields,
    render_markdown,
)

NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}

MOCK_PROPOSAL = Path(__file__).parent.parent / "files" / "mock_proposal.pdf"
MOCK_LEGAL    = Path(__file__).parent.parent / "files" / "mock_document.pdf"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _make_table_pdf(rows: list[tuple[str, str]]) -> str:
    """Write a minimal 2-column bordered-table PDF and return its path."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    col1, col2, row_h = 70, 110, 8
    for name, value in rows:
        pdf.cell(col1, row_h, name,  border=1)
        pdf.cell(col2, row_h, value, border=1, **NL)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    pdf.output(tmp.name)
    tmp.close()
    return tmp.name


def _field(name="Interest Rate", value="SORA + 2.50%", page=1) -> FieldEntry:
    return FieldEntry(name=name, value=value, page=page)


def _make_vectorstore(section_key="3.1", heading="Interest Rate", score=0.9):
    """Return a mock InMemoryVectorStore whose top-1 result is the given section."""
    doc = Document(
        page_content=heading,
        metadata={"section_key": section_key, "heading": heading, "full_text": f"{section_key} {heading}\nSome legal text."},
    )
    vs = MagicMock()
    vs.asimilarity_search_with_score = AsyncMock(return_value=[(doc, score)])
    return vs


def _make_match_chain(match: bool, reason: str = "They agree."):
    chain = MagicMock()
    chain.ainvoke = AsyncMock(return_value=FieldMatch(match=match, reason=reason))
    return chain


# ─────────────────────────────────────────────
# TestIsSectionLabel
# ─────────────────────────────────────────────

class TestIsSectionLabel:
    def test_section_prefix_detected(self):
        assert _is_section_label("SECTION A:  BORROWER DETAILS") is True

    def test_part_prefix_detected(self):
        assert _is_section_label("PART I") is True

    def test_normal_field_name_not_label(self):
        assert _is_section_label("Interest Rate") is False

    def test_short_all_caps_allowed(self):
        # Short all-caps like "LTV" is a valid field name, not a section label
        assert _is_section_label("LTV") is False

    def test_all_caps_long_no_separator_is_label(self):
        assert _is_section_label("BORROWER DETAILS") is True

    def test_mixed_case_field_not_label(self):
        assert _is_section_label("Governing Law") is False


# ─────────────────────────────────────────────
# TestExtractHeading
# ─────────────────────────────────────────────

class TestExtractHeading:
    def test_strips_section_number(self):
        assert _extract_heading("3.1", "3.1 Interest Rate\nSome text.") == "Interest Rate"

    def test_strips_top_level_number(self):
        assert _extract_heading("2", "2. THE FACILITY\nText.") == "THE FACILITY"

    def test_preamble_key_returns_fixed_label(self):
        assert _extract_heading("preamble", "Anything") == "Agreement Parties Preamble"

    def test_empty_text_falls_back_to_key(self):
        assert _extract_heading("5.2", "") == "5.2"

    def test_strips_deep_section_number(self):
        assert _extract_heading("3.3.1", "3.3.1 DSCR Covenant\nText.") == "DSCR Covenant"


# ─────────────────────────────────────────────
# TestMatchLabel
# ─────────────────────────────────────────────

class TestMatchLabel:
    def test_true_returns_match(self):
        assert _match_label(True) == "Match"

    def test_false_returns_deviation(self):
        assert _match_label(False) == "Deviation"

    def test_none_returns_unmatched(self):
        assert _match_label(None) == "Unmatched"


# ─────────────────────────────────────────────
# TestExtractFormFields
# ─────────────────────────────────────────────

class TestExtractFormFields:
    def test_reads_two_column_table(self):
        path = _make_table_pdf([
            ("Interest Rate", "SORA + 2.50%"),
            ("Loan Tenor",    "5 years"),
        ])
        try:
            fields = extract_form_fields(path)
            names = [f.name for f in fields]
            assert "Interest Rate" in names
            assert "Loan Tenor"    in names
        finally:
            Path(path).unlink(missing_ok=True)

    def test_page_number_recorded(self):
        path = _make_table_pdf([("DSCR", "1.25x")])
        try:
            fields = extract_form_fields(path)
            assert fields[0].page == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_empty_rows_skipped(self):
        path = _make_table_pdf([
            ("",              ""),
            ("Facility Type", "Term Loan"),
        ])
        try:
            fields = extract_form_fields(path)
            assert all(f.name for f in fields)
            assert all(f.value for f in fields)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_blank_pdf_returns_empty(self):
        pdf = FPDF()
        pdf.add_page()
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        pdf.output(tmp.name)
        tmp.close()
        try:
            assert extract_form_fields(tmp.name) == []
        finally:
            Path(tmp.name).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# TestCheckField
# ─────────────────────────────────────────────

class TestCheckField:
    _sections = {"3.1": "3.1 Interest Rate\nInterest shall accrue at SORA + 2.50% per annum."}

    async def test_match_path(self):
        result = await check_field(
            field=_field("Interest Rate", "SORA + 2.50%"),
            vectorstore=_make_vectorstore("3.1", "Interest Rate", score=0.9),
            sections=self._sections,
            match_chain=_make_match_chain(match=True, reason="Values agree."),
        )
        assert result.match is True
        assert result.section_key == "3.1"
        assert result.reason == "Values agree."

    async def test_deviation_path(self):
        result = await check_field(
            field=_field("Interest Rate", "SORA + 3.50%"),
            vectorstore=_make_vectorstore("3.1", "Interest Rate", score=0.9),
            sections=self._sections,
            match_chain=_make_match_chain(match=False, reason="Rate differs."),
        )
        assert result.match is False
        assert result.reason == "Rate differs."

    async def test_low_confidence_returns_unmatched(self):
        result = await check_field(
            field=_field("Governing Law", "Singapore"),
            vectorstore=_make_vectorstore("6.1", "Non-Payment", score=0.2),
            sections={"6.1": "6.1 Non-Payment\nText."},
            match_chain=_make_match_chain(match=True),
        )
        assert result.match is None
        assert "No closely matching section" in result.reason

    async def test_empty_vectorstore_returns_unmatched(self):
        vs = MagicMock()
        vs.asimilarity_search_with_score = AsyncMock(return_value=[])
        result = await check_field(
            field=_field(),
            vectorstore=vs,
            sections={},
            match_chain=_make_match_chain(match=True),
        )
        assert result.match is None
        assert result.section_key == "—"

    async def test_chain_error_returns_unmatched(self):
        chain = MagicMock()
        chain.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        result = await check_field(
            field=_field(),
            vectorstore=_make_vectorstore(score=0.9),
            sections={"3.1": "3.1 Interest Rate\nText."},
            match_chain=chain,
        )
        assert result.match is None
        assert "Comparison error" in result.reason

    async def test_score_at_threshold_passes(self):
        """A score exactly at SCORE_THRESHOLD should proceed to comparison."""
        result = await check_field(
            field=_field(),
            vectorstore=_make_vectorstore(score=SCORE_THRESHOLD),
            sections={"3.1": "3.1 Interest Rate\nText."},
            match_chain=_make_match_chain(match=True),
        )
        assert result.match is True

    async def test_score_below_threshold_blocked(self):
        result = await check_field(
            field=_field(),
            vectorstore=_make_vectorstore(score=SCORE_THRESHOLD - 0.01),
            sections={"3.1": "3.1 Interest Rate\nText."},
            match_chain=_make_match_chain(match=True),
        )
        assert result.match is None


# ─────────────────────────────────────────────
# TestRenderMarkdown
# ─────────────────────────────────────────────

class TestRenderMarkdown:

    def _results(self) -> list[CheckResult]:
        return [
            CheckResult(field=_field("Interest Rate", "SORA + 2.50%"), section_key="3.1",
                        section_heading="Interest Rate", match=True,  reason="Agree."),
            CheckResult(field=_field("DSCR",          "1.20x"),        section_key="3.3",
                        section_heading="Financial Covenants", match=False, reason="Legal says 1.25x."),
            CheckResult(field=_field("Governing Law",  "Singapore"),   section_key="—",
                        section_heading="—", match=None, reason="No match found."),
        ]

    def test_summary_counts_correct(self):
        md = render_markdown("proposal.pdf", "legal.pdf", self._results())
        assert "| 1 | 1 | 1 | 3 |" in md

    def test_deviation_label_present(self):
        md = render_markdown("proposal.pdf", "legal.pdf", self._results())
        assert "Deviation" in md

    def test_match_label_present(self):
        md = render_markdown("proposal.pdf", "legal.pdf", self._results())
        assert "Match" in md

    def test_unmatched_label_present(self):
        md = render_markdown("proposal.pdf", "legal.pdf", self._results())
        assert "Unmatched" in md

    def test_file_names_in_header(self):
        md = render_markdown("files/proposal.pdf", "files/legal.pdf", self._results())
        assert "proposal.pdf" in md
        assert "legal.pdf"    in md


# ─────────────────────────────────────────────
# TestIntegration -- requires both mock PDFs
# ─────────────────────────────────────────────

_both_pdfs = pytest.mark.skipif(
    not (MOCK_PROPOSAL.exists() and MOCK_LEGAL.exists()),
    reason="Mock PDFs not found -- run generate_mock_proposal.py and generate_mock_pdf.py first",
)


@_both_pdfs
class TestIntegration:

    def test_all_expected_fields_extracted(self):
        fields = extract_form_fields(str(MOCK_PROPOSAL))
        names = {f.name for f in fields}
        for expected in ["Interest Rate", "Loan Tenor", "DSCR Covenant",
                         "Prepayment Fee", "Governing Law", "Facility Amount"]:
            assert expected in names, f"Missing field: {expected}"

    def test_deviation_fields_present(self):
        """The two deliberate deviations must be in the extracted fields."""
        fields = extract_form_fields(str(MOCK_PROPOSAL))
        by_name = {f.name: f.value for f in fields}
        assert "Prepayment Fee" in by_name
        assert "1.50%" in by_name["Prepayment Fee"]
        assert "DSCR Covenant" in by_name
        assert "1.20" in by_name["DSCR Covenant"]

    def test_legal_document_sections_parsed(self):
        from parse_sections import extract_blocks, parse_sections
        sections = parse_sections(extract_blocks(str(MOCK_LEGAL)))
        for key in ["1", "2", "3", "3.1", "3.3", "4.1"]:
            assert key in sections, f"Missing section: {key}"

    def test_heading_extraction_on_legal_sections(self):
        from parse_sections import extract_blocks, parse_sections
        sections = parse_sections(extract_blocks(str(MOCK_LEGAL)))
        heading_31 = _extract_heading("3.1", sections["3.1"])
        assert "Interest" in heading_31
