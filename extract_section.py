"""
Extract a named section from a PDF.

Python API
----------
    from pipeline import DocumentIndex
    from pipeline.extract import run_extract

    index = DocumentIndex.from_pdf("agreement.pdf")
    result = run_extract(index, "Schedule 2 Compounded Rate Terms", verify=True)
    print(result.heading, result.pages)
    print(result.content)

CLI
---
    # Build index + extract (one-shot)
    python extract_section.py --pdf agreement.pdf --query "3.1 Purpose"

    # With LLM verification and retry
    python extract_section.py --pdf agreement.pdf --query "Schedule 2" --verify
"""

import argparse

from pipeline import DocumentIndex
from pipeline.extract import run_extract


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="extract_section",
        description="Extract a named section from a PDF using LLM-assisted semantic search.",
    )
    parser.add_argument("--pdf",   required=True, help="Path to the PDF file.")
    parser.add_argument("--query", required=True, help='Section to find, e.g. "3.1 Purpose".')
    parser.add_argument("--verify", action="store_true",
                        help="Run LLM verification with retry loop.")
    parser.add_argument("--llm-model",       default="gemma3-27b-it", metavar="MODEL")
    parser.add_argument("--embedding-model", default="bge-m3",        metavar="MODEL")
    parser.add_argument("--base-url",        default="http://localhost:11434/v1", metavar="URL")
    args = parser.parse_args()

    index = DocumentIndex.from_pdf(
        pdf_path=args.pdf,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        embedding_model=args.embedding_model,
    )

    result = run_extract(
        index=index,
        section_query=args.query,
        verify=args.verify,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
    )

    print("\n── Extraction Result ──────────────────────────────────────────")
    print(f"Section  : {result.heading}")
    print(f"Pages    : {result.pages}")
    print(f"Verified : {result.verified}")
    print(f"\n{result.content[:1000]}")


if __name__ == "__main__":
    main()
