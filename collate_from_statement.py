"""
Collate evidence for a free-text statement from a PDF.

Python API
----------
    from pipeline import DocumentIndex
    from pipeline.collate import run_collate

    index = DocumentIndex.from_pdf("agreement.pdf")
    result = run_collate(
        index,
        statement="Extension Option: the Final Maturity Date can be extended...",
        synthesize_result=True,
    )
    print(result.combined_context)
    print(result.synthesis)

CLI
---
    python collate_from_statement.py \\
        --pdf agreement.pdf \\
        --statement "Extension Option: the Final Maturity Date can be extended..."

    # With LLM synthesis
    python collate_from_statement.py \\
        --pdf agreement.pdf \\
        --statement "..." \\
        --synthesize
"""

import argparse

from pipeline import DocumentIndex
from pipeline.collate import run_collate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="collate_from_statement",
        description=(
            "Extract and collate all document content relevant to a free-text "
            "statement, pulling from multiple sections."
        ),
    )
    parser.add_argument("--pdf",       required=True, help="Path to the PDF file.")
    parser.add_argument("--statement", required=True,
                        help="Free-text statement referencing terms across the document.")
    parser.add_argument("--synthesize", action="store_true",
                        help="Run LLM synthesis pass after collation.")
    parser.add_argument("--llm-model",       default="gemma3-27b-it", metavar="MODEL")
    parser.add_argument("--embedding-model", default="bge-m3",        metavar="MODEL")
    parser.add_argument("--base-url",        default="http://localhost:11434/v1", metavar="URL")
    parser.add_argument("--top-k",      type=int,   default=5,    metavar="N",
                        help="Chunks retrieved per term (default: 5).")
    parser.add_argument("--threshold",  type=float, default=0.35, metavar="F",
                        help="Min similarity score to include a chunk (default: 0.35).")
    parser.add_argument("--max-chunks", type=int,   default=25,   metavar="N",
                        help="Max chunks in collated context (default: 25).")
    args = parser.parse_args()

    index = DocumentIndex.from_pdf(
        pdf_path=args.pdf,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        embedding_model=args.embedding_model,
    )

    result = run_collate(
        index=index,
        statement=args.statement,
        synthesize_result=args.synthesize,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        top_k_per_term=args.top_k,
        score_threshold=args.threshold,
        max_chunks=args.max_chunks,
    )

    print("\n── Collation Result ───────────────────────────────────────────")
    print(f"Terms extracted  : {result.terms_extracted}")
    print(f"Chunks matched   : {result.match_count}")
    print(f"Sections touched : {result.sections_referenced}")
    print(f"\n{result.combined_context}")

    if result.synthesis:
        print("\n── LLM Synthesis ──────────────────────────────────────────────")
        print(result.synthesis)


if __name__ == "__main__":
    main()
