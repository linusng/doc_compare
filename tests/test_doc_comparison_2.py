"""
Unit tests for doc_comparison_2.py

Run with:
    uv run pytest tests/ -v
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import fitz  # PyMuPDF
import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from doc_comparison_2 import (
    _W,
    _page_breaks_in_paragraph,
    _truncate,
    compare_clause,
    load_pdf_as_documents,
    render_markdown,
    stream_deviation_comments,
    BestMatchResult,
    ClauseMatchDecision,
    ComparisonReport,
    ComparisonResult,
    DeviationClassification,
    DeviationItem,
    GroundingVerification,
    MaterialityAssessment,
    SectionLabel,
    TargetedContext,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal in-memory PDF with one text block per page."""
    doc = fitz.Document()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    return doc.tobytes()


def _make_elem(tag: str, attribs: dict | None = None) -> MagicMock:
    """Return a mock XML element with the given tag and optional attributes."""
    elem = MagicMock()
    elem.tag = tag
    _attribs = attribs or {}
    elem.get.side_effect = lambda k, default=None: _attribs.get(k, default)
    return elem


def _make_para_mock(elements: list[MagicMock]) -> MagicMock:
    """Return a mock docx paragraph whose ._p.iter() yields the given elements."""
    para = MagicMock()
    para._p.iter.return_value = iter(elements)
    return para


def _deviation_item(
    item_no: int = 1,
    deviation: bool = False,
    score: int = 0,
    severity: str = "N/A",
    deviation_type: str = "none",
    section: str = "Interest Rate",
    base_page: int = 1,
    compare_page: int = 1,
    base_paragraph: str = "Base text.",
    compare_paragraph: str = "Compare text.",
    comments: str | None = None,
) -> DeviationItem:
    return DeviationItem(
        item_no=item_no,
        section=section,
        base_page=base_page,
        compare_page=compare_page,
        base_paragraph=base_paragraph,
        compare_paragraph=compare_paragraph,
        deviation=deviation,
        deviation_type=deviation_type,
        score=score,
        severity=severity,
        comments=comments,
    )


def _make_chains(
    needs_lookup: bool = True,
    best_index: int = 0,
    confidence: str = "High",
    has_deviation: bool = True,
    deviation_type: str = "modification",
    section: str = "Interest Rate",
    grounded: bool = True,
    grounding_issues: list[str] | None = None,
    materiality_score: int = 3,
    materiality_severity: str = "Minor",
    targeted_snippet: str = "Relevant base snippet.",
) -> dict:
    """Return a chains dict where every .ainvoke is an AsyncMock."""

    async def clause_match_invoke(_):
        return ClauseMatchDecision(needs_lookup=needs_lookup, reason="test")

    async def best_match_invoke(_):
        return BestMatchResult(best_index=best_index, confidence=confidence)

    async def targeted_context_invoke(_):
        return TargetedContext(snippet=targeted_snippet)

    async def deviation_class_invoke(_):
        return DeviationClassification(
            has_deviation=has_deviation, deviation_type=deviation_type
        )

    async def section_label_invoke(_):
        return SectionLabel(section=section)

    async def grounding_invoke(_):
        return GroundingVerification(
            grounded=grounded, issues=grounding_issues or []
        )

    async def materiality_invoke(_):
        return MaterialityAssessment(
            score=materiality_score, severity=materiality_severity
        )

    def _mock(fn):
        m = MagicMock()
        m.ainvoke = AsyncMock(side_effect=fn)
        return m

    return {
        "clause_match": _mock(clause_match_invoke),
        "best_match": _mock(best_match_invoke),
        "targeted_context": _mock(targeted_context_invoke),
        "deviation_classification": _mock(deviation_class_invoke),
        "section_label": _mock(section_label_invoke),
        "grounding": _mock(grounding_invoke),
        "materiality": _mock(materiality_invoke),
    }


def _make_vectorstore(docs: list[Document] | None = None) -> MagicMock:
    vs = MagicMock()
    vs.asimilarity_search = AsyncMock(return_value=docs or [])
    return vs


def _make_openai_client(reply: str = "The interest rate was changed.") -> MagicMock:
    """Return a mock AsyncOpenAI client whose stream yields reply as single chunk."""

    async def _stream_iter():
        chunk = MagicMock()
        chunk.choices[0].delta.content = reply
        yield chunk

    completions = MagicMock()
    completions.create = AsyncMock(return_value=_stream_iter())
    client = MagicMock()
    client.chat.completions = completions
    return client


# ─────────────────────────────────────────────────────────────────────────────
# TestTruncate
# ─────────────────────────────────────────────────────────────────────────────

class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_text_exactly_at_max_unchanged(self):
        text = "a" * 250
        assert _truncate(text, 250) == text

    def test_long_text_gets_ellipsis(self):
        result = _truncate("a" * 300)
        assert result.endswith("…")
        assert len(result) <= 251  # 250 chars + ellipsis

    def test_newlines_replaced_with_spaces(self):
        result = _truncate("line one\nline two", 100)
        assert "\n" not in result
        assert "line one line two" == result

    def test_leading_trailing_whitespace_stripped(self):
        assert _truncate("  hello  ", 100) == "hello"


# ─────────────────────────────────────────────────────────────────────────────
# TestPydanticModels
# ─────────────────────────────────────────────────────────────────────────────

class TestPydanticModels:
    def test_materiality_score_below_1_raises(self):
        with pytest.raises(ValidationError):
            MaterialityAssessment(score=0, severity="Cosmetic")

    def test_materiality_score_above_5_raises(self):
        with pytest.raises(ValidationError):
            MaterialityAssessment(score=6, severity="Critical")

    def test_deviation_item_defaults(self):
        item = _deviation_item()
        assert item.deviation is False
        assert item.deviation_type == "none"
        assert item.score == 0
        assert item.severity == "N/A"
        assert item.comments is None

    def test_comparison_result_deviation_count_counts_only_true(self):
        items = [
            _deviation_item(item_no=1, deviation=True),
            _deviation_item(item_no=2, deviation=False),
            _deviation_item(item_no=3, deviation=True),
        ]
        result = ComparisonResult(deviations=items)
        assert result.deviation_count == 2

    def test_best_match_result_accepts_negative_index(self):
        # best_index=-1 is the sentinel for "no match"
        r = BestMatchResult(best_index=-1, confidence="Low")
        assert r.best_index == -1


# ─────────────────────────────────────────────────────────────────────────────
# TestPageBreaks
# ─────────────────────────────────────────────────────────────────────────────

class TestPageBreaks:
    def test_no_breaks_returns_zero(self):
        para = _make_para_mock([_make_elem("{ns}w")])
        assert _page_breaks_in_paragraph(para) == 0

    def test_last_rendered_page_break_counted(self):
        elem = _make_elem("{someNs}lastRenderedPageBreak")
        para = _make_para_mock([elem])
        assert _page_breaks_in_paragraph(para) == 1

    def test_explicit_page_break_w_br_counted(self):
        elem = _make_elem(
            f"{{{_W}}}br",
            {f"{{{_W}}}type": "page"},
        )
        para = _make_para_mock([elem])
        assert _page_breaks_in_paragraph(para) == 1

    def test_multiple_mixed_breaks_summed(self):
        elems = [
            _make_elem("{ns}lastRenderedPageBreak"),
            _make_elem(f"{{{_W}}}br", {f"{{{_W}}}type": "page"}),
            _make_elem("{ns}lastRenderedPageBreak"),
        ]
        para = _make_para_mock(elems)
        assert _page_breaks_in_paragraph(para) == 3


# ─────────────────────────────────────────────────────────────────────────────
# TestRenderMarkdown
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderMarkdown:
    def _report(self, items: list[DeviationItem]) -> ComparisonReport:
        return ComparisonReport(
            base_doc="base.pdf",
            compare_doc="offer.docx",
            result=ComparisonResult(deviations=items),
        )

    def test_empty_deviations_shows_placeholder(self):
        md = render_markdown(self._report([]))
        assert "_No clauses examined._" in md

    def test_deviation_count_in_header(self):
        items = [_deviation_item(deviation=True), _deviation_item(item_no=2)]
        md = render_markdown(self._report(items))
        assert "**Material deviations found:** 1" in md

    def test_deviation_row_shows_yes(self):
        item = _deviation_item(
            deviation=True,
            deviation_type="modification",
            score=4,
            severity="Major",
            comments="Rate was lowered.",
        )
        md = render_markdown(self._report([item]))
        assert "| Yes |" in md
        assert "4/5 (Major)" in md
        assert "Rate was lowered." in md

    def test_no_deviation_row_shows_no(self):
        md = render_markdown(self._report([_deviation_item(deviation=False)]))
        assert "| No |" in md

    def test_zero_score_materiality_shows_dash(self):
        item = _deviation_item(score=0, deviation=False)
        md = render_markdown(self._report([item]))
        # score=0 → materiality cell should be "—"
        assert "| — |" in md


# ─────────────────────────────────────────────────────────────────────────────
# TestLoadPdfAsDocuments
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadPdfAsDocuments:
    def _write_pdf(self, pages: list[str]) -> str:
        pdf_bytes = _make_pdf_bytes(pages)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        tmp.write(pdf_bytes)
        tmp.close()
        return tmp.name

    def test_single_page_returns_one_document(self):
        path = self._write_pdf(["Hello world"])
        try:
            docs = load_pdf_as_documents(path)
            assert len(docs) == 1
            assert "Hello world" in docs[0].page_content
            assert docs[0].metadata["page"] == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_multi_page_returns_one_doc_per_page(self):
        path = self._write_pdf(["Page one text", "Page two text", "Page three text"])
        try:
            docs = load_pdf_as_documents(path)
            assert len(docs) == 3
            pages = [d.metadata["page"] for d in docs]
            assert pages == [1, 2, 3]
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_doc_name_stored_in_metadata(self):
        path = self._write_pdf(["Some text"])
        try:
            docs = load_pdf_as_documents(path, doc_name="custom_name.pdf")
            assert docs[0].metadata["doc_name"] == "custom_name.pdf"
        finally:
            Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# TestStreamDeviationComments
# ─────────────────────────────────────────────────────────────────────────────

class TestStreamDeviationComments:
    async def test_chunks_assembled_into_string(self):
        async def _stream():
            for word in ["The ", "rate ", "changed."]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = word
                yield chunk

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_stream())

        result = await stream_deviation_comments(
            openai_client=client,
            model="test-model",
            base_clause="Base clause.",
            compare_clause="Compare clause.",
            deviation_type="modification",
        )
        assert result == "The rate changed."

    async def test_correction_notes_appear_in_prompt(self):
        captured_prompt = {}

        async def _stream():
            chunk = MagicMock()
            chunk.choices[0].delta.content = "Fixed."
            yield chunk

        async def _create(**kwargs):
            captured_prompt["content"] = kwargs["messages"][0]["content"]
            return _stream()

        client = MagicMock()
        client.chat.completions.create = AsyncMock(side_effect=_create)

        await stream_deviation_comments(
            openai_client=client,
            model="test-model",
            base_clause="Base.",
            compare_clause="Compare.",
            deviation_type="omission",
            correction_notes=["Claim X is unsupported"],
        )
        assert "Claim X is unsupported" in captured_prompt["content"]

    async def test_none_delta_content_handled(self):
        """Chunks with delta.content=None should not crash (treated as empty)."""

        async def _stream():
            for content in [None, "Answer."]:
                chunk = MagicMock()
                chunk.choices[0].delta.content = content
                yield chunk

        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=_stream())

        result = await stream_deviation_comments(
            openai_client=client,
            model="test-model",
            base_clause="B.",
            compare_clause="C.",
            deviation_type="addition",
        )
        assert result == "Answer."


# ─────────────────────────────────────────────────────────────────────────────
# TestCompareClause
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareClause:
    """Tests for compare_clause() covering every major branch."""

    def _compare_doc(self, text: str = "The interest rate shall be 5%.", page: int = 2) -> Document:
        return Document(page_content=text, metadata={"page": page})

    def _base_doc(self, text: str = "The interest rate shall be 4%.", page: int = 1) -> Document:
        return Document(page_content=text, metadata={"page": page})

    async def _run(self, compare_doc, vectorstore, chains, client, model="m"):
        return await compare_clause(
            compare_doc=compare_doc,
            item_no=1,
            vectorstore=vectorstore,
            openai_client=client,
            model=model,
            chains=chains,
            semaphore=asyncio.Semaphore(5),
        )

    async def test_boilerplate_clause_skipped(self):
        """needs_lookup=False → return immediately with no deviation."""
        chains = _make_chains(needs_lookup=False)
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is False
        assert result.comments is None
        # Downstream chains should never have been called
        chains["best_match"].ainvoke.assert_not_called()

    async def test_empty_vectorstore_returns_no_deviation(self):
        """No retrieved candidates → no deviation."""
        chains = _make_chains(needs_lookup=True)
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(docs=[]),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is False
        chains["best_match"].ainvoke.assert_not_called()

    async def test_low_confidence_match_returns_no_deviation(self):
        """BestMatchResult.confidence='Low' → no deviation."""
        chains = _make_chains(needs_lookup=True, best_index=0, confidence="Low")
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(docs=[self._base_doc()]),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is False
        chains["deviation_classification"].ainvoke.assert_not_called()

    async def test_no_deviation_path_returns_section_label(self):
        """has_deviation=False → DeviationItem with deviation=False but section filled."""
        chains = _make_chains(
            needs_lookup=True,
            best_index=0,
            confidence="High",
            has_deviation=False,
            section="Repayment Terms",
        )
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(docs=[self._base_doc()]),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is False
        assert result.section == "Repayment Terms"
        chains["grounding"].ainvoke.assert_not_called()

    @patch("doc_comparison_2.stream_deviation_comments", new_callable=AsyncMock)
    async def test_full_deviation_path(self, mock_stream):
        """has_deviation=True → DeviationItem fully populated."""
        mock_stream.return_value = "Rate dropped from 4% to 5%."
        chains = _make_chains(
            needs_lookup=True,
            best_index=0,
            confidence="High",
            has_deviation=True,
            deviation_type="modification",
            section="Interest Rate",
            grounded=True,
            materiality_score=4,
            materiality_severity="Major",
        )
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(docs=[self._base_doc()]),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is True
        assert result.deviation_type == "modification"
        assert result.section == "Interest Rate"
        assert result.score == 4
        assert result.severity == "Major"
        assert result.comments == "Rate dropped from 4% to 5%."
        assert mock_stream.call_count == 1  # no retry needed

    @patch("doc_comparison_2.stream_deviation_comments", new_callable=AsyncMock)
    async def test_grounding_failure_triggers_retry(self, mock_stream):
        """grounding.grounded=False → stream_deviation_comments called twice."""
        mock_stream.return_value = "Corrected explanation."
        chains = _make_chains(
            needs_lookup=True,
            best_index=0,
            confidence="High",
            has_deviation=True,
            deviation_type="omission",
            grounded=False,
            grounding_issues=["Claim about page 3 is unsupported"],
            materiality_score=3,
            materiality_severity="Minor",
        )
        result = await self._run(
            compare_doc=self._compare_doc(),
            vectorstore=_make_vectorstore(docs=[self._base_doc()]),
            chains=chains,
            client=_make_openai_client(),
        )
        assert result.deviation is True
        assert mock_stream.call_count == 2
        # Second call should include correction_notes
        _, second_kwargs = mock_stream.call_args_list[1]
        assert second_kwargs.get("correction_notes") == ["Claim about page 3 is unsupported"]
