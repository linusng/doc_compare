"""
Find the passage of content that best matches a FREE-TEXT query in a PDF.

Like extract_section.py, but instead of resolving a named section it searches
the document content for the passage closest to an arbitrary string. If nothing
matches, it prints "No".

Python API
----------
    from pipeline import DocumentIndex
    from pipeline.find_content import run_find

    index = DocumentIndex.from_pdf("agreement.pdf")
    result = run_find(index, "the borrower must maintain insurance", verify=True)
    if result.found:
        print(result.heading, result.pages)
        print(result.content)
    else:
        print("No")

CLI
---
    python find_content.py --pdf agreement.pdf \
        --query "lender may accelerate the loan on default"

    # Return the closest passage without LLM verification
    python find_content.py --pdf agreement.pdf --query "..." --no-verify
"""

import argparse

from pipeline import DocumentIndex
from pipeline.find_content import run_find


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="find_content",
        description="Find the passage of content closest to a free-text query in a PDF.",
    )
    parser.add_argument("--pdf",   required=True, help="Path to the PDF file.")
    parser.add_argument("--query", required=True, help="Free-text to search for.")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="Skip LLM verification; return the closest passage.")
    parser.add_argument("--top-k",     type=int,   default=8, metavar="N")
    parser.add_argument("--min-score", type=float, default=None, metavar="F",
                        help="Similarity floor; below it the result is No.")
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

    result = run_find(
        index=index,
        query=args.query,
        verify=args.verify,
        top_k=args.top_k,
        min_score=args.min_score,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
    )

    print("\n── Find Result ────────────────────────────────────────────────")
    print(f"Match    : {result.answer}")
    if not result.found:
        print("No")
        return

    print(f"Confirmed: {len(result.matches)} match(es)")
    for i, m in enumerate(result.matches, start=1):
        tag = " (expanded to section body)" if m.expanded else ""
        print(f"\n[{i}] score={m.score:.4f}  pages={m.pages}  {m.heading!r}{tag}")
        print(m.content[:1000])


if __name__ == "__main__":
    main()
