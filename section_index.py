"""
Section / Schedule Vector Index
================================
Indexes the dict[str, str] output of parse_sections.py or parse_schedule.py
into a LangChain InMemoryVectorStore so you can retrieve the most relevant
sections for any query.

Each key-value pair from the parser becomes one Document:
    page_content = section text
    metadata     = {"key": "1.1", "source": "sections"}

Two retrieval functions are provided:
    query_store            → list[Document]           (top-k by similarity)
    query_store_with_score → list[tuple[Document, float]]  (top-k + cosine score)

Typical usage
-------------
    from parse_sections import extract_blocks, parse_sections
    from section_index  import build_store, make_embeddings, query_store_with_score

    embeddings = make_embeddings()          # reads OPENAI_API_KEY from env
    blocks     = extract_blocks("loan.pdf")
    sections   = parse_sections(blocks)
    store      = build_store(sections, embeddings, source="sections")

    hits = query_store_with_score(store, "financial covenants DSCR", k=3)
    for doc, score in hits:
        print(doc.metadata["key"], round(score, 4))
        print(doc.page_content[:200])

Combining sections and schedules
---------------------------------
    from parse_schedule_2 import parse_schedules_v2

    schedules  = parse_schedules_v2(blocks)
    store      = build_store({**sections, **schedules}, embeddings, source="combined")
"""

import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings


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
        Optional custom endpoint for OpenAI-compatible APIs (e.g. Azure,
        local proxies).  Mirrors the ``--base-url`` flag in doc_comparison.py.
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
    sections: dict[str, str],
    embeddings: Embeddings,
    source: str = "",
) -> InMemoryVectorStore:
    """
    Embed every key-value pair and load them into an InMemoryVectorStore.

    Parameters
    ----------
    sections:
        Dict of {section_key: section_text} as returned by parse_sections
        or parse_schedules.
    embeddings:
        Any LangChain Embeddings instance (e.g. from make_embeddings()).
    source:
        Label stored in each Document's metadata under "source".  Useful
        when you later merge sections and schedules into one store and need
        to distinguish them in results.

    Returns
    -------
    InMemoryVectorStore
        Ready for similarity search.
    """
    docs = [
        Document(
            page_content=text.strip(),
            metadata={"key": key, "source": source},
        )
        for key, text in sections.items()
        if text.strip()
    ]
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
    """
    Return the top-k most relevant Documents for the query.

    Parameters
    ----------
    store:
        A store built with build_store().
    query:
        Free-text query.
    k:
        Number of results to return.
    """
    return store.similarity_search(query, k=k)


def query_store_with_score(
    store: InMemoryVectorStore,
    query: str,
    k: int = 5,
) -> list[tuple[Document, float]]:
    """
    Return the top-k most relevant Documents together with their similarity
    scores (higher = more similar).

    Parameters
    ----------
    store:
        A store built with build_store().
    query:
        Free-text query.
    k:
        Number of results to return.
    """
    return store.similarity_search_with_score(query, k=k)


# ─────────────────────────────────────────────
# I/O helper
# ─────────────────────────────────────────────

def load_json(path: str | Path) -> dict[str, str]:
    """Load a sections/schedules JSON file saved by the parser CLI."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Build an InMemoryVectorStore from a parsed sections JSON file "
            "and run a similarity search query against it."
        )
    )
    parser.add_argument(
        "json_file",
        help="Path to the JSON file produced by any parse_sections / parse_schedules script",
    )
    parser.add_argument(
        "query",
        help="Free-text query to search for",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)",
    )
    parser.add_argument(
        "--scores",
        action="store_true",
        help="Print the similarity score alongside each result",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="Embedding model name (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=300,
        help="Number of content characters to preview per result (default: 300)",
    )
    args = parser.parse_args()

    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading sections from {json_path.name} ...", flush=True)
    sections = load_json(json_path)
    print(f"  {len(sections)} sections loaded.", flush=True)

    print("Building vector store ...", flush=True)
    embeddings = make_embeddings(
        model=args.model,
        base_url=args.base_url,
    )
    store = build_store(sections, embeddings, source=json_path.stem)
    print(f"  Store ready ({len(sections)} documents).", flush=True)

    print(f'\nQuery: "{args.query}"')
    print(f"Top {args.k} results:\n" + "─" * 60)

    if args.scores:
        results = query_store_with_score(store, args.query, k=args.k)
        for rank, (doc, score) in enumerate(results, start=1):
            preview = doc.page_content[: args.preview].replace("\n", " ")
            print(f"[{rank}] {doc.metadata['key']}  (score: {score:.4f})")
            print(f"    {preview}")
            print()
    else:
        results = query_store(store, args.query, k=args.k)
        for rank, doc in enumerate(results, start=1):
            preview = doc.page_content[: args.preview].replace("\n", " ")
            print(f"[{rank}] {doc.metadata['key']}")
            print(f"    {preview}")
            print()
