"""
Unit tests for section_index_2.py

All tests use _FakeEmbeddings — no real API calls.
"""

import hashlib
import math
import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from section_index_2 import (
    _DEFAULT_CHUNK_OVERLAP,
    _DEFAULT_CHUNK_SIZE,
    _base_key,
    build_store,
    chunk_sections,
    deduplicate_results,
    deduplicate_results_with_score,
    load_json,
    query_store,
    query_store_with_score,
)


# ─────────────────────────────────────────────
# Fake embeddings
# ─────────────────────────────────────────────

class _FakeEmbeddings(Embeddings):
    DIM = 16

    def _embed(self, text: str) -> list[float]:
        raw  = hashlib.md5(text.encode()).digest()
        vec  = [(b / 127.5) - 1.0 for b in raw[: self.DIM]]
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
def sample_sections() -> dict[str, dict]:
    return {
        "1":   {"text": "1. DEFINITIONS\nDefined terms follow.",        "page_numbers": [1]},
        "1.1": {"text": "1.1 Definitions\nCapitalised terms.",          "page_numbers": [1, 2]},
        "2":   {"text": "2. THE FACILITY\nTerm loan facility.",         "page_numbers": [3]},
        "3.3": {"text": "3.3 Financial Covenants\nDSCR >= 1.25x.",      "page_numbers": [8, 9]},
    }


@pytest.fixture
def store(sample_sections, embeddings):
    return build_store(sample_sections, embeddings, source="sections")


# ─────────────────────────────────────────────
# TestBuildStore
# ─────────────────────────────────────────────

class TestBuildStore:
    def test_document_count_matches_section_count(self, sample_sections, embeddings):
        store   = build_store(sample_sections, embeddings)
        results = query_store(store, "anything", k=100)
        assert len(results) == len(sample_sections)

    def test_metadata_key_preserved(self, store, sample_sections):
        results     = query_store(store, "anything", k=100)
        returned    = {doc.metadata["key"] for doc in results}
        assert returned == set(sample_sections.keys())

    def test_metadata_source_label(self, sample_sections, embeddings):
        store   = build_store(sample_sections, embeddings, source="sections")
        results = query_store(store, "anything", k=100)
        assert all(doc.metadata["source"] == "sections" for doc in results)

    def test_metadata_page_numbers_present(self, store):
        results = query_store(store, "anything", k=100)
        for doc in results:
            assert "page_numbers" in doc.metadata
            assert isinstance(doc.metadata["page_numbers"], list)

    def test_page_numbers_match_input(self, store, sample_sections):
        results    = query_store(store, "anything", k=100)
        result_map = {doc.metadata["key"]: doc.metadata["page_numbers"] for doc in results}
        for key, data in sample_sections.items():
            assert result_map[key] == data["page_numbers"]

    def test_single_page_number_stored(self, embeddings):
        sections = {"1": {"text": "Content.", "page_numbers": [5]}}
        store    = build_store(sections, embeddings)
        results  = query_store(store, "anything", k=1)
        assert results[0].metadata["page_numbers"] == [5]

    def test_multi_page_numbers_stored(self, embeddings):
        sections = {"2": {"text": "Content.", "page_numbers": [3, 4, 5]}}
        store    = build_store(sections, embeddings)
        results  = query_store(store, "anything", k=1)
        assert results[0].metadata["page_numbers"] == [3, 4, 5]

    def test_page_content_matches_text_field(self, store, sample_sections):
        results    = query_store(store, "anything", k=100)
        result_map = {doc.metadata["key"]: doc.page_content for doc in results}
        for key, data in sample_sections.items():
            assert result_map[key] == data["text"].strip()

    def test_empty_text_excluded(self, embeddings):
        sections = {
            "1": {"text": "   ", "page_numbers": [1]},
            "2": {"text": "Real content.", "page_numbers": [2]},
        }
        store   = build_store(sections, embeddings)
        results = query_store(store, "anything", k=10)
        keys    = {doc.metadata["key"] for doc in results}
        assert "1" not in keys
        assert "2" in keys

    def test_missing_page_numbers_defaults_to_empty_list(self, embeddings):
        sections = {"1": {"text": "Content."}}   # no page_numbers key
        store    = build_store(sections, embeddings)
        results  = query_store(store, "anything", k=1)
        assert results[0].metadata["page_numbers"] == []

    def test_empty_sections_returns_empty_store(self, embeddings):
        store   = build_store({}, embeddings)
        results = query_store(store, "anything", k=5)
        assert results == []

    def test_combined_sections_and_schedules(self, embeddings):
        sections  = {
            "1":          {"text": "Definitions.", "page_numbers": [1]},
        }
        schedules = {
            "Schedule 1": {"text": "CP text.",     "page_numbers": [20]},
        }
        store   = build_store({**sections, **schedules}, embeddings, source="combined")
        results = query_store(store, "anything", k=10)
        assert len(results) == 2
        pg_map  = {doc.metadata["key"]: doc.metadata["page_numbers"] for doc in results}
        assert pg_map["1"]          == [1]
        assert pg_map["Schedule 1"] == [20]


# ─────────────────────────────────────────────
# TestQueryStore
# ─────────────────────────────────────────────

class TestQueryStore:
    def test_returns_list_of_documents(self, store):
        results = query_store(store, "covenants", k=2)
        assert isinstance(results, list)
        assert all(isinstance(d, Document) for d in results)

    def test_k_limits_results(self, store):
        for k in (1, 2, 3):
            assert len(query_store(store, "test", k=k)) <= k

    def test_each_result_has_page_numbers(self, store):
        for doc in query_store(store, "definitions", k=4):
            assert "page_numbers" in doc.metadata


# ─────────────────────────────────────────────
# TestQueryStoreWithScore
# ─────────────────────────────────────────────

class TestQueryStoreWithScore:
    def test_returns_tuples_with_float_score(self, store):
        for doc, score in query_store_with_score(store, "facility", k=2):
            assert isinstance(doc, Document)
            assert isinstance(score, float)

    def test_scores_descending(self, store):
        results = query_store_with_score(store, "definitions", k=4)
        scores  = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_has_page_numbers(self, store):
        for doc, _ in query_store_with_score(store, "covenants", k=3):
            assert "page_numbers" in doc.metadata


# ─────────────────────────────────────────────
# TestChunkSections
# ─────────────────────────────────────────────

class TestChunkSections:
    def test_short_section_unchanged(self):
        data     = {"1.1": {"text": "Short.", "page_numbers": [2]}}
        result   = chunk_sections(data, chunk_size=1000)
        assert result == data

    def test_long_section_is_split(self):
        long_text = "word " * 5000
        data      = {"1": {"text": long_text, "page_numbers": [1, 2]}}
        result    = chunk_sections(data, chunk_size=10_000, chunk_overlap=100)
        assert len(result) > 1
        assert all("1 [" in k for k in result)

    def test_chunks_inherit_page_numbers(self):
        long_text = "word " * 5000
        data      = {"3.3": {"text": long_text, "page_numbers": [8, 9]}}
        result    = chunk_sections(data, chunk_size=10_000, chunk_overlap=0)
        for chunk_data in result.values():
            assert chunk_data["page_numbers"] == [8, 9]

    def test_chunk_keys_format(self):
        long_text = "a " * 6000
        data      = {"3.3": {"text": long_text, "page_numbers": [5]}}
        result    = chunk_sections(data, chunk_size=10_000, chunk_overlap=0)
        for key in result:
            assert key.startswith("3.3 [") and key.endswith("]")

    def test_chunk_text_fields_are_strings(self):
        long_text = "b " * 6000
        data      = {"2": {"text": long_text, "page_numbers": [3]}}
        result    = chunk_sections(data, chunk_size=10_000, chunk_overlap=0)
        for chunk_data in result.values():
            assert isinstance(chunk_data["text"], str)
            assert isinstance(chunk_data["page_numbers"], list)

    def test_only_long_sections_are_split(self):
        data = {
            "short": {"text": "Brief.",        "page_numbers": [1]},
            "long":  {"text": "word " * 5000, "page_numbers": [2, 3]},
        }
        result = chunk_sections(data, chunk_size=10_000, chunk_overlap=0)
        assert "short" in result
        assert "long"  not in result
        assert any("long [" in k for k in result)

    def test_empty_sections_unchanged(self):
        assert chunk_sections({}) == {}


# ─────────────────────────────────────────────
# TestDeduplication
# ─────────────────────────────────────────────

def _doc(key: str, pages: list[int] | None = None) -> Document:
    return Document(
        page_content="text",
        metadata={"key": key, "source": "test", "page_numbers": pages or []},
    )


class TestDeduplicateResults:
    def test_no_chunks_unchanged(self):
        docs = [_doc("1"), _doc("2"), _doc("3")]
        assert deduplicate_results(docs) == docs

    def test_duplicate_chunks_collapsed(self):
        docs   = [_doc("3.3 [1/3]"), _doc("3.3 [2/3]"), _doc("3.3 [3/3]")]
        result = deduplicate_results(docs)
        assert len(result) == 1
        assert result[0].metadata["key"] == "3.3 [1/3]"

    def test_page_numbers_preserved_on_surviving_chunk(self):
        docs   = [_doc("3.3 [1/2]", [8, 9]), _doc("3.3 [2/2]", [8, 9])]
        result = deduplicate_results(docs)
        assert result[0].metadata["page_numbers"] == [8, 9]

    def test_empty_input(self):
        assert deduplicate_results([]) == []


class TestDeduplicateResultsWithScore:
    def test_highest_scoring_chunk_kept(self):
        pairs = [
            (_doc("3.3 [1/3]", [8]), 0.75),
            (_doc("3.3 [2/3]", [9]), 0.92),
            (_doc("3.3 [3/3]", [9]), 0.60),
        ]
        result     = deduplicate_results_with_score(pairs)
        doc, score = result[0]
        assert doc.metadata["key"] == "3.3 [2/3]"
        assert score == 0.92

    def test_page_numbers_on_winning_chunk(self):
        pairs = [
            (_doc("1 [1/2]", [1, 2]), 0.70),
            (_doc("1 [2/2]", [1, 2]), 0.85),
        ]
        result     = deduplicate_results_with_score(pairs)
        doc, _     = result[0]
        assert doc.metadata["page_numbers"] == [1, 2]

    def test_result_sorted_descending(self):
        pairs = [
            (_doc("1"), 0.70),
            (_doc("2"), 0.95),
            (_doc("3"), 0.60),
        ]
        result = deduplicate_results_with_score(pairs)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_empty_input(self):
        assert deduplicate_results_with_score([]) == []


# ─────────────────────────────────────────────
# TestLoadJson
# ─────────────────────────────────────────────

class TestLoadJson:
    def test_roundtrip(self, tmp_path):
        data = {
            "1":   {"text": "Section one.", "page_numbers": [1]},
            "1.1": {"text": "Sub one.",     "page_numbers": [1, 2]},
        }
        path = tmp_path / "sections.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert load_json(path) == data

    def test_accepts_string_path(self, tmp_path):
        data = {"Schedule 1": {"text": "CP.", "page_numbers": [20]}}
        path = tmp_path / "schedules.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        assert load_json(str(path)) == data
