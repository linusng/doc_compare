"""
Section / Schedule Vector Index v2
====================================
Indexes the dict[str, dict] output of parse_sections_3.py or
parse_schedules_2.py into a LangChain InMemoryVectorStore.

Compared to section_index.py this version accepts the nested JSON format
produced by the page-number-aware parsers:

    Input (section_index.py):
        {"1.1": "section text..."}

    Input (section_index_2.py):
        {"1.1": {"text": "section text...", "page_numbers": [2, 3]}}

Each Document's metadata carries the page numbers alongside the key and
source label, so every retrieved result is immediately traceable back to
the source document:

    metadata = {
        "key":          "1.1",
        "source":       "sections",
        "page_numbers": [2, 3],
    }

Typical usage
-------------
    from parse_sections_3 import extract_blocks, parse_sections
    from section_index_2  import build_store, make_embeddings, query_store_with_score

    embeddings = make_embeddings()
    blocks     = extract_blocks("loan.pdf")
    sections   = parse_sections(blocks)
    store      = build_store(sections, embeddings, source="sections")

    hits = query_store_with_score(store, "financial covenants DSCR", k=3)
    for doc, score in hits:
        print(doc.metadata["key"], doc.metadata["page_numbers"], round(score, 4))
        print(doc.page_content[:200])

Combining sections and schedules
---------------------------------
    from parse_schedules_2 import extract_blocks_second_half, parse_schedules

    sched_blocks = extract_blocks_second_half("loan.pdf")
    schedules    = parse_schedules(sched_blocks)
    store        = build_store({**sections, **schedules}, embeddings, source="combined")
"""

import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─────────────────────────────────────────────
# Embeddings factory
# ─────────────────────────────────────────────

def make_embeddings(
    model: str = "text-embedding-3-small",
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAIEmbeddings:
    """
    Return an OpenAIEmbeddings instance.

    Parameters
    ----------
    model:
        Embedding model name.  Defaults to ``text-embedding-3-small``.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` environment
        variable when not provided.
    base_url:
        Optional custom endpoint for OpenAI-compatible APIs.
    """
    kwargs: dict = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAIEmbeddings(**kwargs)


# ─────────────────────────────────────────────
# Store builder
# ─────────────────────────────────────────────

def build_store(
    sections: dict[str, dict],
    embeddings: Embeddings,
    source: str = "",
) -> InMemoryVectorStore:
    """
    Embed every section and load the documents into an InMemoryVectorStore.

    Parameters
    ----------
    sections:
        Nested dict ``{section_key: {"text": ..., "page_numbers": [...]}}``
        as returned by ``parse_sections_3.parse_sections`` or
        ``parse_schedules_2.parse_schedules``.
    embeddings:
        Any LangChain Embeddings instance (e.g. from ``make_embeddings()``).
    source:
        Label stored in each Document's metadata under ``"source"``.

    Returns
    -------
    InMemoryVectorStore
        Ready for similarity search.  Each Document carries:
            metadata["key"]          – section key  (e.g. "1.1")
            metadata["source"]       – the ``source`` argument
            metadata["page_numbers"] – list of 1-based page numbers
    """
    docs = []
    for key, data in sections.items():
        text = data["text"].strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "key":          key,
                    "source":       source,
                    "page_numbers": data.get("page_numbers", []),
                },
            )
        )
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(docs)
    return store


# ─────────────────────────────────────────────
# Query helpers
# ─────────────────────────────────────────────

def query_store(
    store: InMemoryVectorStore,
    query: str,
    k: int = 5,
) -> list[Document]:
    """Return the top-k most relevant Documents for the query."""
    return store.similarity_search(query, k=k)


def query_store_with_score(
    store: InMemoryVectorStore,
    query: str,
    k: int = 5,
) -> list[tuple[Document, float]]:
    """Return the top-k most relevant Documents with similarity scores."""
    return store.similarity_search_with_score(query, k=k)


# ─────────────────────────────────────────────
# Chunking
# ─────────────────────────────────────────────

_DEFAULT_CHUNK_SIZE    = 21_000
_DEFAULT_CHUNK_OVERLAP =    800


def _base_key(key: str) -> str:
    """Strip the chunk suffix from a key if present.

    "3.3 [2/4]"  →  "3.3"
    "Schedule 1" →  "Schedule 1"
    """
    return re.sub(r'\s+\[\d+/\d+\]$', '', key)


def chunk_sections(
    sections: dict[str, dict],
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> dict[str, dict]:
    """
    Split any section whose text exceeds ``chunk_size`` characters into
    overlapping chunks.

    Each chunk inherits the parent section's ``page_numbers`` — precise
    per-chunk page attribution is not possible at the character level, so
    the full set of pages for the original section is preserved on every
    chunk for traceability.

    Parameters
    ----------
    sections:
        Nested dict ``{key: {"text": ..., "page_numbers": [...]}}``
    chunk_size:
        Maximum characters per chunk (default 21,000 ≈ 6,000 BGE-M3 tokens).
    chunk_overlap:
        Character overlap between consecutive chunks (default 800).

    Returns
    -------
    dict[str, dict]
        Expanded mapping; long sections replaced by numbered chunk entries,
        each carrying the original ``page_numbers``.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    result: dict[str, dict] = {}
    for key, data in sections.items():
        text         = data["text"]
        page_numbers = data.get("page_numbers", [])

        if len(text) <= chunk_size:
            result[key] = data
        else:
            chunks = splitter.split_text(text)
            if len(chunks) == 1:
                result[key] = data
            else:
                total = len(chunks)
                for i, chunk in enumerate(chunks, start=1):
                    result[f"{key} [{i}/{total}]"] = {
                        "text":         chunk,
                        "page_numbers": page_numbers,   # inherited from parent
                    }
    return result


# ─────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────

def deduplicate_results(results: list[Document]) -> list[Document]:
    """
    Collapse multiple chunk hits from the same parent section into one,
    keeping the first (highest-ranked) occurrence.
    """
    seen:   set[str]       = set()
    unique: list[Document] = []
    for doc in results:
        base = _base_key(doc.metadata["key"])
        if base not in seen:
            seen.add(base)
            unique.append(doc)
    return unique


def deduplicate_results_with_score(
    results: list[tuple[Document, float]],
) -> list[tuple[Document, float]]:
    """
    Collapse multiple chunk hits from the same parent section into one,
    keeping the chunk with the highest similarity score, then re-sort
    descending.
    """
    best: dict[str, tuple[Document, float]] = {}
    for doc, score in results:
        base = _base_key(doc.metadata["key"])
        if base not in best or score > best[base][1]:
            best[base] = (doc, score)
    return sorted(best.values(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
# I/O helper
# ─────────────────────────────────────────────

def load_json(path: str | Path) -> dict[str, dict]:
    """Load a nested sections/schedules JSON file saved by the parser CLI."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Build an InMemoryVectorStore from a paged sections JSON file "
            "and run a similarity search query against it."
        )
    )
    parser.add_argument(
        "json_file",
        help=(
            "Path to the JSON file produced by parse_sections_3.py or "
            "parse_schedules_2.py (nested format with page_numbers)"
        ),
    )
    parser.add_argument("query", help="Free-text query to search for")
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of results to return after deduplication (default: 5)",
    )
    parser.add_argument(
        "--scores", action="store_true",
        help="Print the similarity score alongside each result",
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Custom OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--preview", type=int, default=300,
        help="Number of content characters to preview per result (default: 300)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE,
        help=f"Max characters per chunk (default: {_DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=_DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap characters between chunks (default: {_DEFAULT_CHUNK_OVERLAP})",
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    # ── 1. Load ───────────────────────────────────────────────────────────
    print(f"Loading sections from {json_path.name} ...", flush=True)
    sections = load_json(json_path)
    print(f"  {len(sections)} sections loaded.", flush=True)

    # ── 2. Chunk long sections ────────────────────────────────────────────
    chunked = chunk_sections(
        sections,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    n_split = len(chunked) - len(sections)
    if n_split > 0:
        print(f"  {n_split} extra chunk(s) created for sections exceeding "
              f"{args.chunk_size:,} chars.", flush=True)
    else:
        print("  All sections within chunk size limit — no splitting needed.",
              flush=True)

    # ── 3. Build store ────────────────────────────────────────────────────
    print("Building vector store ...", flush=True)
    embeddings = make_embeddings(model=args.model, base_url=args.base_url)
    store = build_store(chunked, embeddings, source=json_path.stem)
    print(f"  Store ready ({len(chunked)} document(s)).", flush=True)

    # ── 4. Query & deduplicate ────────────────────────────────────────────
    fetch_k = args.k * 4

    print(f'\nQuery: "{args.query}"')
    print(f"Top {args.k} results (after deduplication):\n" + "─" * 60)

    if args.scores:
        raw     = query_store_with_score(store, args.query, k=fetch_k)
        results = deduplicate_results_with_score(raw)[: args.k]
        for rank, (doc, score) in enumerate(results, start=1):
            base  = _base_key(doc.metadata["key"])
            pages = doc.metadata.get("page_numbers", [])
            preview = doc.page_content[: args.preview].replace("\n", " ")
            print(f"[{rank}] {base}  (score: {score:.4f})  pages: {pages}")
            print(f"    {preview}")
            print()
    else:
        raw     = query_store(store, args.query, k=fetch_k)
        results = deduplicate_results(raw)[: args.k]
        for rank, doc in enumerate(results, start=1):
            base  = _base_key(doc.metadata["key"])
            pages = doc.metadata.get("page_numbers", [])
            preview = doc.page_content[: args.preview].replace("\n", " ")
            print(f"[{rank}] {base}  pages: {pages}")
            print(f"    {preview}")
            print()
