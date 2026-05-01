"""
CP (Conditions Precedent) field checker.

For each field in a CP JSON file, collates relevant FA passages then asks
the LLM whether the FA deviates from what the CP form states.  Results are
returned as a pandas DataFrame.

Input JSON shape (one field per key)
-------------------------------------
{
    "Facility Amount":         "USD 50,000,000",
    "Final Maturity Date":     "3 years from first utilisation",
    "Extension Option":        "Two extensions of 1 year each",
    "Margin":                  "2.50% per annum",
    ...
}

Python API
----------
    from pipeline import DocumentIndex
    from pipeline.cp_checker import run_cp_check

    index = DocumentIndex.from_pdf("facility_agreement.pdf")
    df = run_cp_check(index, "cp_fields.json")
    df.to_excel("cp_review.xlsx", index=False)

CLI
---
    python -m pipeline.cp_checker \\
        --pdf facility_agreement.pdf \\
        --cp  cp_fields.json \\
        --out cp_review.xlsx
"""

import argparse
import json
import re
import sys

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from .collate import (
    build_collated_context,
    extract_key_terms,
    gather_all_relevant,
)
from .ingestion import DocumentIndex


# ── Structured deviation result ───────────────────────────────────────────────

class DeviationAnalysis(BaseModel):
    """Structured output of the per-field deviation LLM call."""
    has_deviation: bool
    comments: str


# ── Deviation analysis ────────────────────────────────────────────────────────

def analyze_deviation(
    cp_field: str,
    cp_value: str,
    combined_context: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
) -> DeviationAnalysis:
    """
    Compare one CP field/value pair against the FA passages retrieved by
    collation and return a structured verdict.

    Returns
    -------
    DeviationAnalysis
        has_deviation : True if the FA materially differs from the CP value.
        comments      : Concise explanation — what matches, what diverges, and
                        which FA clause is the source.
    """
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a legal document analyst comparing a term sheet / "
            "conditions precedent (CP) form against a signed facility agreement (FA).\n\n"
            "You will be given:\n"
            "  CP FIELD   — the name of the term being checked\n"
            "  CP VALUE   — what the CP form states for that term\n"
            "  FA CONTEXT — relevant passages retrieved from the FA\n\n"
            "Your task:\n"
            "1. Determine whether the FA materially deviates from the CP value.\n"
            "   A deviation exists when the FA says something DIFFERENT, MISSING,\n"
            "   or MORE RESTRICTIVE than the CP value.\n"
            "   No deviation if the FA confirms or is consistent with the CP value.\n\n"
            "2. Write a concise comment (1–3 sentences) that:\n"
            "   - States what the FA actually says for this term.\n"
            "   - Identifies the specific deviation (if any), or confirms alignment.\n"
            "   - Cites the FA section/clause where the information was found.\n\n"
            "Respond with ONLY valid JSON in this exact shape:\n"
            '{"has_deviation": true, "comments": "..."}\n'
            "has_deviation must be a boolean (true or false, not a string)."
        )),
        ("human", (
            "CP FIELD  : {cp_field}\n"
            "CP VALUE  : {cp_value}\n\n"
            "FA CONTEXT:\n{fa_context}"
        )),
    ])

    raw = (prompt | llm).invoke({
        "cp_field":  cp_field,
        "cp_value":  cp_value,
        "fa_context": combined_context[:5000],   # cap to avoid context overflow
    }).content.strip()

    # Parse JSON from response (tolerant of surrounding text)
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return DeviationAnalysis(
                has_deviation=bool(data.get("has_deviation", False)),
                comments=str(data.get("comments", "")).strip(),
            )
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: couldn't parse JSON — flag for manual review
    return DeviationAnalysis(
        has_deviation=False,
        comments=f"[Parse error — manual review required] Raw LLM output: {raw[:300]}",
    )


# ── Per-field collation + deviation ──────────────────────────────────────────

def _check_one_field(
    cp_field: str,
    cp_value: str,
    index: DocumentIndex,
    llm_model: str,
    base_url: str,
    api_key: str,
    top_k_per_term: int,
    score_threshold: float,
    max_chunks: int,
) -> dict:
    """
    Run the full collate + deviation pipeline for one CP field.
    Returns a dict matching the DataFrame schema.
    """
    # Build a statement combining the field name and its value so both
    # the term name and the specific figure/condition are searchable.
    statement = f"{cp_field}: {cp_value}"

    # ── Collation (equivalent to run_collate but without the model params
    #    duplicated — reuses the index that was already built) ──────────────
    terms = extract_key_terms(
        statement, llm_model=llm_model, base_url=base_url, api_key=api_key,
    )

    ranked = gather_all_relevant(
        terms, index,
        top_k_per_term=top_k_per_term,
        score_threshold=score_threshold,
    )

    combined_context, evidence, sections_referenced = build_collated_context(
        ranked, max_chunks=max_chunks,
    )

    # ── Deviation analysis ─────────────────────────────────────────────────
    deviation = analyze_deviation(
        cp_field=cp_field,
        cp_value=str(cp_value),
        combined_context=combined_context,
        llm_model=llm_model,
        base_url=base_url,
        api_key=api_key,
    )

    return {
        "CP Field":        cp_field,
        "CP Form Value":   cp_value,
        "FA Legal Section": combined_context,
        "Deviation":       "Yes" if deviation.has_deviation else "No",
        "Comments":        deviation.comments,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_cp_check(
    index: DocumentIndex,
    cp_json_path: str,
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    top_k_per_term: int = 5,
    score_threshold: float = 0.35,
    max_chunks: int = 25,
) -> pd.DataFrame:
    """
    Check every field in the CP JSON against the FA and return a DataFrame.

    Parameters
    ----------
    index         : Pre-built DocumentIndex (call DocumentIndex.from_pdf first).
    cp_json_path  : Path to a JSON file mapping CP field names → CP values.
    llm_model     : Ollama LLM model name.
    base_url      : Ollama base URL.
    api_key       : API key (use "ollama" for local Ollama).
    top_k_per_term: Chunks retrieved per extracted term in collation.
    score_threshold: Min similarity score to include a chunk.
    max_chunks    : Max chunks in collated context per field.

    Returns
    -------
    pd.DataFrame with columns:
        #, CP Field, CP Form Value, FA Legal Section, Deviation, Comments
    """
    with open(cp_json_path, encoding="utf-8") as f:
        data: dict = json.load(f)

    if not data:
        raise ValueError(f"CP JSON file is empty: {cp_json_path}")

    rows = []
    total = len(data)

    for i, (cp_field, cp_value) in enumerate(data.items(), start=1):
        print(f"\n[{i}/{total}] Checking: {cp_field!r}")
        print(f"           CP value : {cp_value}")

        try:
            row = _check_one_field(
                cp_field=cp_field,
                cp_value=cp_value,
                index=index,
                llm_model=llm_model,
                base_url=base_url,
                api_key=api_key,
                top_k_per_term=top_k_per_term,
                score_threshold=score_threshold,
                max_chunks=max_chunks,
            )
        except Exception as exc:
            print(f"           ⚠ Error: {exc}", file=sys.stderr)
            row = {
                "CP Field":        cp_field,
                "CP Form Value":   cp_value,
                "FA Legal Section": "",
                "Deviation":       "No",
                "Comments":        f"[Error during processing: {exc}]",
            }

        print(f"           Deviation: {row['Deviation']}  |  {row['Comments'][:80]}...")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add enumeration as the first column
    df.insert(0, "#", range(1, len(df) + 1))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cp_checker",
        description="Check CP form fields against a facility agreement PDF.",
    )
    parser.add_argument("--pdf",  required=True, help="Path to the FA PDF.")
    parser.add_argument("--cp",   required=True, help="Path to the CP fields JSON file.")
    parser.add_argument("--out",  default="cp_review.xlsx",
                        help="Output file path (.xlsx or .csv). Default: cp_review.xlsx")
    parser.add_argument("--llm-model",       default="gemma3-27b-it", metavar="MODEL")
    parser.add_argument("--embedding-model", default="bge-m3",        metavar="MODEL")
    parser.add_argument("--base-url",        default="http://localhost:11434/v1", metavar="URL")
    parser.add_argument("--top-k",      type=int,   default=5,    metavar="N",
                        help="Chunks per term in collation (default: 5).")
    parser.add_argument("--threshold",  type=float, default=0.35, metavar="F",
                        help="Min similarity score (default: 0.35).")
    parser.add_argument("--max-chunks", type=int,   default=25,   metavar="N",
                        help="Max chunks per field in collated context (default: 25).")
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    print(f"Building document index from: {args.pdf}")
    index = DocumentIndex.from_pdf(
        pdf_path=args.pdf,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        embedding_model=args.embedding_model,
    )

    df = run_cp_check(
        index=index,
        cp_json_path=args.cp,
        llm_model=args.llm_model,
        base_url=args.base_url,
        api_key="ollama",
        top_k_per_term=args.top_k,
        score_threshold=args.threshold,
        max_chunks=args.max_chunks,
    )

    print(f"\n── CP Review Summary ──────────────────────────────────────────")
    print(f"Total fields checked : {len(df)}")
    print(f"Deviations found     : {(df['Deviation'] == 'Yes').sum()}")
    print(f"\n{df[['#', 'CP Field', 'Deviation', 'Comments']].to_string(index=False)}")

    # Save output
    if args.out.endswith(".csv"):
        df.to_csv(args.out, index=False, encoding="utf-8-sig")
    else:
        df.to_excel(args.out, index=False, engine="openpyxl")

    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
