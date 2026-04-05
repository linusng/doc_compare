"""
Unit tests for section_index.py

All tests use _FakeEmbeddings — no real API calls are made.
"""

import hashlib
import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from section_index import build_store, load_json, query_store, query_store_with_score


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
