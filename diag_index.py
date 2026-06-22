"""
Diagnostic: dump the chunk list produced by DocumentIndex.from_pdf so we can see
exactly how a section was split during ingestion and where it cuts off.

Usage
-----
    # Show every chunk (id, heading level, start page, body length, heading)
    python diag_index.py --pdf agreement.pdf

    # Only chunks whose heading OR body contains a phrase (case-insensitive).
    # Use a phrase you KNOW lives in the second half of Definitions to find out
    # which chunk/heading it landed under — or whether it survived ingestion.
    python diag_index.py --pdf agreement.pdf --grep "business day"
"""

import argparse

from pipeline import DocumentIndex


def main() -> None:
    ap = argparse.ArgumentParser(prog="diag_index")
    ap.add_argument("--pdf", required=True, help="Path to the PDF file.")
    ap.add_argument("--grep", default="",
                    help="Only show chunks whose heading/body contains this text.")
    ap.add_argument("--full", action="store_true",
                    help="Also print each shown chunk's full body content.")
    ap.add_argument("--llm-model", default="gemma3-27b-it")
    ap.add_argument("--embedding-model", default="bge-m3")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    args = ap.parse_args()

    index = DocumentIndex.from_pdf(
        pdf_path=args.pdf,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        embedding_model=args.embedding_model,
    )

    needle = args.grep.lower()
    print(f"\nTotal chunks: {len(index.chunks)}\n")
    print(f"{'id':>4}  {'lvl':>3}  {'pg':>3}  {'len':>5}  heading")
    print("-" * 80)

    for c in index.chunks:
        if needle and needle not in c.heading.lower() and needle not in c.content.lower():
            continue
        print(f"{c.chunk_id:>4}  {c.heading_level:>3}  {c.start_page:>3}  "
              f"{len(c.content):>5}  {c.heading[:70]!r}")
        if args.full:
            print("        " + c.content.replace("\n", "\n        "))
            print("-" * 80)


if __name__ == "__main__":
    main()
