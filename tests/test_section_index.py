"""
Unit tests for section_index.py

All tests use _FakeEmbeddings — no real API calls are made.
"""

import hashlib
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from section_index import (
    build_store,
    chunk_sections,
    deduplicate_results,
    deduplicate_results_with_score,
    load_json,
    query_store,
    query_store_with_score,
    _base_key,
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_CHUNK_OVERLAP,
)


# ─────────────────────────────────────────────
# Fake embeddings (no API calls)
# ─────────────────────────────────────────────

class _FakeEmbeddings(Embeddings):
    """
    Deterministic 16-dim embeddings derived from an MD5 hash of the text.
    Vectors are L2-normalised so cosine similarity is well-defined.
    Identical texts produce identical vectors; similar texts may not cluster
    meaningfully, but Document structure and metadata round-trips are fully
    testable without a real embedding model.
    """
    DIM = 16

    def _embed(self, text: str) -> list[float]:
        raw = hashlib.md5(text.encode()).digest()
        vec = [(b / 127.5) - 1.0 for b in raw[: self.DIM]]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings() -> _FakeEmbeddings:
    return _FakeEmbeddings()


@pytest.fixture
def sample_sections() -> dict[str, str]:
    return {
        "1":   "1. DEFINITIONS AND INTERPRETATION\nDefined terms follow.",
        "1.1": "1.1 Definitions\nCapitalised terms have the meanings given here.",
        "2":   "2. THE FACILITY\nThe lender agrees to provide a term loan.",
        "3.3": "3.3 Financial Covenants\nDSCR shall not fall below 1.25x.",
    }


@pytest.fixture
def store(sample_sections, embeddings):
    return build_store(sample_sections, embeddings, source="sections")


# ─────────────────────────────────────────────
# TestBuildStore
# ─────────────────────────────────────────────

class TestBuildStore:
    def test_document_count_matches_section_count(self, sample_sections, embeddings):
        store = build_store(sample_sections, embeddings)
        results = query_store(store, "anything", k=100)
        assert len(results) == len(sample_sections)

    def test_metadata_key_preserved(self, store, sample_sections):
        results = query_store(store, "anything", k=100)
        returned_keys = {doc.metadata["key"] for doc in results}
        assert returned_keys == set(sample_sections.keys())

    def test_metadata_source_label(self, sample_sections, embeddings):
        store = build_store(sample_sections, embeddings, source="sections")
        results = query_store(store, "anything", k=100)
        assert all(doc.metadata["source"] == "sections" for doc in results)

    def test_source_label_schedules(self, embeddings):
        store = build_store({"Schedule 1": "CP text."}, embeddings, source="schedules")
        results = query_store(store, "anything", k=5)
        assert results[0].metadata["source"] == "schedules"

    def test_page_content_matches_section_text(self, store, sample_sections):
        results = query_store(store, "anything", k=100)
        content_map = {doc.metadata["key"]: doc.page_content for doc in results}
        for key, text in sample_sections.items():
            assert content_map[key] == text.strip()

    def test_empty_sections_dict_returns_empty_store(self, embeddings):
        store = build_store({}, embeddings)
        results = query_store(store, "test", k=5)
        assert results == []

    def test_whitespace_only_values_excluded(self, embeddings):
        sections = {"1": "   \n  ", "2": "Real content."}
        store = build_store(sections, embeddings)
        results = query_store(store, "anything", k=10)
        keys = {doc.metadata["key"] for doc in results}
        assert "1" not in keys
        assert "2" in keys

    def test_combined_sections_and_schedules(self, embeddings):
        sections  = {"1": "Definitions.", "2": "Facility."}
        schedules = {"Schedule 1": "Conditions Precedent.", "Schedule 2": "Forms."}
        store = build_store({**sections, **schedules}, embeddings, source="combined")
        results = query_store(store, "anything", k=10)
        assert len(results) == 4
        assert all(doc.metadata["source"] == "combined" for doc in results)


# ─────────────────────────────────────────────
# TestQueryStore
# ─────────────────────────────────────────────

class TestQueryStore:
    def test_returns_list_of_documents(self, store):
        results = query_store(store, "financial covenants", k=2)
        assert isinstance(results, list)
        assert all(isinstance(doc, Document) for doc in results)

    def test_k_limits_results(self, store):
        for k in (1, 2, 3):
            results = query_store(store, "definitions", k=k)
            assert len(results) <= k

    def test_k_larger_than_store_returns_all(self, store, sample_sections):
        results = query_store(store, "definitions", k=100)
        assert len(results) == len(sample_sections)

    def test_exact_match_text_is_top_result(self, embeddings):
        # When the query IS the section text, that section should rank first
        # (fake embeddings are deterministic so identical text → identical vector)
        sections = {"target": "DSCR covenant text.", "other": "Repayment schedule."}
        store = build_store(sections, embeddings)
        results = query_store(store, "DSCR covenant text.", k=2)
        assert results[0].metadata["key"] == "target"


# ─────────────────────────────────────────────
# TestQueryStoreWithScore
# ─────────────────────────────────────────────

class TestQueryStoreWithScore:
    def test_returns_list_of_tuples(self, store):
        results = query_store_with_score(store, "facility", k=2)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_first_element_is_document(self, store):
        results = query_store_with_score(store, "facility", k=2)
        assert all(isinstance(doc, Document) for doc, _ in results)

    def test_second_element_is_float(self, store):
        results = query_store_with_score(store, "facility", k=2)
        assert all(isinstance(score, float) for _, score in results)

    def test_k_limits_results(self, store):
        for k in (1, 2, 3):
            results = query_store_with_score(store, "definitions", k=k)
            assert len(results) <= k

    def test_exact_match_has_highest_score(self, embeddings):
        sections = {"target": "Financial covenants.", "other": "Repayment terms."}
        store = build_store(sections, embeddings)
        results = query_store_with_score(store, "Financial covenants.", k=2)
        assert results[0][0].metadata["key"] == "target"
        assert results[0][1] >= results[1][1]

    def test_scores_descending(self, store):
        results = query_store_with_score(store, "definitions", k=4)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_intact_in_scored_results(self, store):
        results = query_store_with_score(store, "covenants", k=3)
        for doc, _ in results:
            assert "key" in doc.metadata
            assert "source" in doc.metadata


# ─────────────────────────────────────────────
# TestLoadJson
# ─────────────────────────────────────────────

class TestLoadJson:
    def test_roundtrip(self, tmp_path):
        data = {"1": "Section one text.", "1.1": "Subsection text."}
        path = tmp_path / "sections.json"
        path.write_text(__import__("json").dumps(data), encoding="utf-8")
        loaded = load_json(path)
        assert loaded == data

    def test_accepts_string_path(self, tmp_path):
        data = {"Schedule 1": "CP text."}
        path = tmp_path / "schedules.json"
        path.write_text(__import__("json").dumps(data), encoding="utf-8")
        loaded = load_json(str(path))
        assert loaded == data


# ─────────────────────────────────────────────
# TestBaseKey
# ─────────────────────────────────────────────

class TestBaseKey:
    def test_chunk_suffix_stripped(self):
        assert _base_key("3.3 [2/4]") == "3.3"

    def test_first_chunk_suffix_stripped(self):
        assert _base_key("Schedule 1 [1/3]") == "Schedule 1"

    def test_no_suffix_unchanged(self):
        assert _base_key("3.3") == "3.3"
        assert _base_key("Schedule 1") == "Schedule 1"
        assert _base_key("preamble") == "preamble"

    def test_single_chunk_suffix_stripped(self):
        assert _base_key("1 [1/1]") == "1"


# ─────────────────────────────────────────────
# TestChunkSections
# ─────────────────────────────────────────────

class TestChunkSections:
    def test_short_section_unchanged(self):
        sections = {"1.1": "Short text that fits easily."}
        result = chunk_sections(sections, chunk_size=1000)
        assert result == {"1.1": "Short text that fits easily."}

    def test_long_section_is_split(self):
        long_text = "word " * 5000           # ~25,000 chars
        sections  = {"1": long_text}
        result    = chunk_sections(sections, chunk_size=10_000, chunk_overlap=100)
        assert len(result) > 1
        assert all("1 [" in k for k in result)

    def test_chunk_keys_format(self):
        long_text = "a " * 6000
        sections  = {"3.3": long_text}
        result    = chunk_sections(sections, chunk_size=10_000, chunk_overlap=0)
        for key in result:
            assert key.startswith("3.3 [")
            assert "/" in key
            assert key.endswith("]")

    def test_chunk_numbering_sequential(self):
        long_text = "b " * 8000
        sections  = {"2": long_text}
        result    = chunk_sections(sections, chunk_size=10_000, chunk_overlap=0)
        total = len(result)
        for i in range(1, total + 1):
            assert f"2 [{i}/{total}]" in result

    def test_all_content_present_across_chunks(self):
        # Every word in the original must appear somewhere in the chunks
        words     = [f"word{i}" for i in range(500)]
        long_text = " ".join(words)
        sections  = {"sec": long_text}
        result    = chunk_sections(sections, chunk_size=1_000, chunk_overlap=50)
        combined  = " ".join(result.values())
        for word in words:
            assert word in combined

    def test_multiple_sections_only_long_ones_split(self):
        sections = {
            "short": "Brief text.",
            "long":  "word " * 5000,
        }
        result = chunk_sections(sections, chunk_size=10_000, chunk_overlap=0)
        assert "short" in result           # unchanged
        assert "long" not in result        # replaced by chunks
        assert any("long [" in k for k in result)

    def test_empty_sections_unchanged(self):
        assert chunk_sections({}) == {}

    def test_default_chunk_size_constant_is_reasonable(self):
        # Default should be well under 8192 * 4 chars (≈32,768)
        assert _DEFAULT_CHUNK_SIZE < 32_768
        assert _DEFAULT_CHUNK_OVERLAP < _DEFAULT_CHUNK_SIZE


# ─────────────────────────────────────────────
# TestDeduplicateResults
# ─────────────────────────────────────────────

def _doc(key: str, content: str = "text") -> Document:
    return Document(page_content=content, metadata={"key": key, "source": "test"})


class TestDeduplicateResults:
    def test_no_chunks_unchanged(self):
        docs = [_doc("1"), _doc("2"), _doc("3")]
        assert deduplicate_results(docs) == docs

    def test_duplicate_chunks_collapsed_to_first(self):
        docs = [_doc("3.3 [1/3]"), _doc("3.3 [2/3]"), _doc("3.3 [3/3]")]
        result = deduplicate_results(docs)
        assert len(result) == 1
        assert result[0].metadata["key"] == "3.3 [1/3]"

    def test_mixed_chunked_and_plain_deduped(self):
        docs = [
            _doc("1.1 [1/2]"),
            _doc("2"),
            _doc("1.1 [2/2]"),   # duplicate of first
            _doc("3"),
        ]
        result = deduplicate_results(docs)
        keys = [d.metadata["key"] for d in result]
        assert len(result) == 3
        assert "1.1 [1/2]" in keys
        assert "1.1 [2/2]" not in keys   # second chunk suppressed
        assert "2" in keys
        assert "3" in keys

    def test_order_preserved(self):
        docs = [_doc("3"), _doc("1"), _doc("2")]
        result = deduplicate_results(docs)
        assert [d.metadata["key"] for d in result] == ["3", "1", "2"]

    def test_empty_input(self):
        assert deduplicate_results([]) == []


class TestDeduplicateResultsWithScore:
    def test_no_chunks_unchanged(self):
        pairs = [(_doc("1"), 0.9), (_doc("2"), 0.8)]
        result = deduplicate_results_with_score(pairs)
        assert len(result) == 2

    def test_highest_scoring_chunk_kept(self):
        pairs = [
            (_doc("3.3 [1/3]"), 0.75),
            (_doc("3.3 [2/3]"), 0.92),   # highest for this section
            (_doc("3.3 [3/3]"), 0.60),
        ]
        result = deduplicate_results_with_score(pairs)
        assert len(result) == 1
        doc, score = result[0]
        assert doc.metadata["key"] == "3.3 [2/3]"
        assert score == 0.92

    def test_result_sorted_by_score_descending(self):
        pairs = [
            (_doc("1 [1/2]"), 0.70),
            (_doc("2"),       0.95),
            (_doc("1 [2/2]"), 0.80),   # higher chunk of "1"
            (_doc("3"),       0.60),
        ]
        result = deduplicate_results_with_score(pairs)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)
        # "1" should be represented by its 0.80 chunk
        keys = [d.metadata["key"] for d, _ in result]
        assert "1 [2/2]" in keys
        assert "1 [1/2]" not in keys

    def test_empty_input(self):
        assert deduplicate_results_with_score([]) == []
