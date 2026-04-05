"""
Form vs Legal Document Check
==============================
Cross-checks every field in a credit proposal form (PDF with Field|Value
tables) against the corresponding section of a facility agreement PDF.

Pipeline per field:
    1. Extract Field|Value rows from proposal PDF via PyMuPDF table detection
       (falls back to LLM extraction if no tables are found)
    2. Parse facility agreement into sections via parse_sections.py
    3. Build an InMemoryVectorStore from section headings
    4. Route each field to its best-matching section (embedding similarity)
    5. Single LLM chain call: does the form value agree with the section?
    6. Render Markdown report

Usage:
    python form_check.py \\
        --form  files/mock_proposal.pdf \\
        --legal files/mock_document.pdf \\
        --base-url http://your-endpoint/v1 \\
        --api-key your-key
"""

import argparse
import asyncio
import os
import re
from pathlib import Path
from typing import Optional

import httpx
import pymupdf
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from parse_sections import extract_blocks, parse_sections


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────

class FieldEntry(BaseModel):
    name: str
    value: str
    page: int = 1


class FormFieldList(BaseModel):
    fields: list[FieldEntry] = Field(
        description="All field name / value pairs extracted from the credit proposal."
    )


class FieldMatch(BaseModel):
    match: bool = Field(
        description=(
            "True if the credit proposal value is consistent with what the legal "
            "document stipulates. False if the values conflict or differ materially."
        )
    )
    reason: str = Field(
        description="One sentence explaining whether they agree or how they differ."
    )


class CheckResult(BaseModel):
    field: FieldEntry
    section_key: str
    section_heading: str
    match: Optional[bool]   # None = not routed / error
    reason: str


# ─────────────────────────────────────────────
# Form field extraction
# ─────────────────────────────────────────────

def _is_section_label(text: str) -> bool:
    """True for section-band headers that are not field rows."""
    t = text.strip()
    return (
        t.upper().startswith("SECTION ")
        or t.upper().startswith("PART ")
        # All-caps line with no colon → likely a heading, not a field name
        or (len(t) > 4 and t == t.upper() and ":" not in t and "/" not in t)
    )


def extract_form_fields(pdf_path: str) -> list[FieldEntry]:
    """
    Primary extraction: PyMuPDF table detection.
    Reads 2-column Field|Value tables from every page.
    """
    fields: list[FieldEntry] = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            for tab in page.find_tables().tables:
                for row in tab.extract():
                    if len(row) < 2:
                        continue
                    name  = (row[0] or "").strip()
                    value = (row[1] or "").strip()
                    if not name or not value:
                        continue
                    if _is_section_label(name):
                        continue
                    fields.append(FieldEntry(name=name, value=value, page=page_no))
    return fields


async def extract_form_fields_llm(
    pdf_path: str,
    llm: ChatOpenAI,
) -> list[FieldEntry]:
    """
    Fallback: ask the LLM to extract fields from raw page text.
    Used when table detection finds nothing (e.g. borderless layout).
    """
    parts: list[str] = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    raw = "\n".join(parts)[:6000]

    chain = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract every form field name and its corresponding value from the "
            "credit proposal text below.\n"
            "Skip section headers, titles, signature lines, and blank entries.\n\n"
            "TEXT:\n{text}"
        ),
    ) | llm.with_structured_output(FormFieldList)

    result: FormFieldList = await chain.ainvoke({"text": raw})
    return result.fields


# ─────────────────────────────────────────────
# Section store
# ─────────────────────────────────────────────

def _extract_heading(key: str, text: str) -> str:
    """
    Strip the section number from the first line of a section to get
    the pure topic words.
    "3.1 Interest Rate\\n..." → "Interest Rate"
    """
    if key == "preamble":
        return "Agreement Parties Preamble"
    first_line = text.splitlines()[0].strip() if text else key
    stripped = re.sub(r'^\d+(\.\d+)*\.?\s*', '', first_line).strip()
    return stripped or first_line


async def build_section_store(
    sections: dict[str, str],
    embeddings: OpenAIEmbeddings,
) -> InMemoryVectorStore:
    """
    Embed section headings (topic words only) for clean semantic routing.
    Full section text is stored separately in Document metadata.
    """
    docs: list[Document] = []
    for key, text in sections.items():
        heading = _extract_heading(key, text)
        docs.append(Document(
            page_content=heading,
            metadata={
                "section_key": key,
                "heading":     heading,
                "full_text":   text,
            },
        ))
    store = InMemoryVectorStore(embedding=embeddings)
    await store.aadd_documents(docs)
    return store


# ─────────────────────────────────────────────
# Per-field check
# ─────────────────────────────────────────────

# Cosine similarity below this → too dissimilar to trust routing
SCORE_THRESHOLD = 0.45
# Cap section text sent to the LLM to avoid token overrun
MAX_SECTION_CHARS = 2000


async def check_field(
    field: FieldEntry,
    vectorstore: InMemoryVectorStore,
    sections: dict[str, str],
    match_chain,
) -> CheckResult:

    # ── Route: find the best-matching section heading ─────────────────────
    try:
        hits = await vectorstore.asimilarity_search_with_score(field.name, k=1)
    except Exception:
        # asimilarity_search_with_score unavailable in this build — fall back
        docs = await vectorstore.asimilarity_search(field.name, k=1)
        hits = [(docs[0], 1.0)] if docs else []

    if not hits:
        return CheckResult(
            field=field,
            section_key="—",
            section_heading="—",
            match=None,
            reason="No sections found in the legal document.",
        )

    best_doc, score = hits[0]
    section_key     = best_doc.metadata["section_key"]
    section_heading = best_doc.metadata["heading"]
    section_text    = sections.get(section_key, "")[:MAX_SECTION_CHARS]

    if score < SCORE_THRESHOLD:
        return CheckResult(
            field=field,
            section_key=section_key,
            section_heading=section_heading,
            match=None,
            reason=(
                f"No closely matching section found "
                f"(best match: '{section_heading}', similarity={score:.2f})."
            ),
        )

    # ── Compare: does the form value agree with the section? ─────────────
    try:
        result: FieldMatch = await match_chain.ainvoke({
            "field_name":   field.name,
            "field_value":  field.value,
            "section_text": section_text,
        })
        return CheckResult(
            field=field,
            section_key=section_key,
            section_heading=section_heading,
            match=result.match,
            reason=result.reason,
        )
    except Exception as exc:
        return CheckResult(
            field=field,
            section_key=section_key,
            section_heading=section_heading,
            match=None,
            reason=f"Comparison error: {exc}",
        )


# ─────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────

def _match_label(match: Optional[bool]) -> str:
    if match is True:
        return "Match"
    if match is False:
        return "Deviation"
    return "Unmatched"


def render_markdown(
    proposal_path: str,
    legal_path: str,
    results: list[CheckResult],
) -> str:
    n_match     = sum(1 for r in results if r.match is True)
    n_deviation = sum(1 for r in results if r.match is False)
    n_unmatched = sum(1 for r in results if r.match is None)

    header = (
        "| # | Field | Form Value | Legal Section | Result | Notes |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = []
    for i, r in enumerate(results, start=1):
        sec = f"{r.section_key} {r.section_heading}" if r.section_key != "—" else "—"
        rows.append(
            f"| {i} "
            f"| {r.field.name} "
            f"| {r.field.value} "
            f"| {sec} "
            f"| {_match_label(r.match)} "
            f"| {r.reason} |\n"
        )

    return (
        f"# Credit Proposal vs Facility Agreement\n\n"
        f"**Proposal:** `{Path(proposal_path).name}`  \n"
        f"**Legal document:** `{Path(legal_path).name}`  \n\n"
        f"| Match | Deviation | Unmatched | Total |\n"
        f"|---|---|---|---|\n"
        f"| {n_match} | {n_deviation} | {n_unmatched} | {len(results)} |\n\n"
        f"---\n\n"
        f"## Field-by-Field Results\n\n"
        f"{header}{''.join(rows)}\n"
        f"---\n"
        f"*Generated by form_check.py*\n"
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-check a credit proposal form against a facility agreement."
    )
    parser.add_argument("--form",        required=True,               help="Credit proposal PDF")
    parser.add_argument("--legal",       required=True,               help="Facility agreement PDF")
    parser.add_argument("--output",      default="form_check_report.md")
    parser.add_argument("--base-url",    required=True,               help="LLM + embeddings API base URL")
    parser.add_argument("--api-key",     default=None)
    parser.add_argument("--model",       default="gpt-oss-120b")
    parser.add_argument("--embed-model", default="bge-m3")
    parser.add_argument("--concurrency", default=5, type=int)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "no-key")

    async with httpx.AsyncClient(
        headers={"api-key": api_key},
        timeout=httpx.Timeout(120.0),
    ):
        langchain_llm = ChatOpenAI(
            openai_api_base=args.base_url,
            openai_api_key=api_key,
            model_name=args.model,
            temperature=0,
            streaming=False,
        )
        embeddings = OpenAIEmbeddings(
            openai_api_base=args.base_url,
            openai_api_key=api_key,
            model=args.embed_model,
        )

        # ── Match chain ──────────────────────────────────────────────────
        match_chain = PromptTemplate(
            input_variables=["field_name", "field_value", "section_text"],
            template=(
                "You are a legal document reviewer.\n\n"
                "The credit proposal states:\n"
                "  {field_name}: {field_value}\n\n"
                "The relevant section of the facility agreement reads:\n"
                "{section_text}\n\n"
                "Does the credit proposal value agree with what the legal document "
                "stipulates for this topic?\n"
                "True  = the values are consistent or within the permitted terms.\n"
                "False = the values conflict, differ materially, or the legal document "
                "stipulates something different."
            ),
        ) | langchain_llm.with_structured_output(FieldMatch)

        # ── Parse legal document ─────────────────────────────────────────
        print(f"Parsing legal document: {args.legal}")
        sections = parse_sections(extract_blocks(args.legal))
        print(f"  -> {len(sections)} sections found")

        # ── Build section store ──────────────────────────────────────────
        print("Building section index...")
        store = await build_section_store(sections, embeddings)
        print(f"  -> {len(sections)} headings indexed")

        # ── Extract form fields ──────────────────────────────────────────
        print(f"\nExtracting fields from: {args.form}")
        fields = extract_form_fields(args.form)
        if not fields:
            print("  Table detection found nothing -- falling back to LLM extraction")
            fields = await extract_form_fields_llm(args.form, langchain_llm)
        print(f"  -> {len(fields)} fields extracted")

        # ── Run checks in parallel ───────────────────────────────────────
        print(f"\nChecking {len(fields)} fields (concurrency={args.concurrency})...")
        semaphore = asyncio.Semaphore(args.concurrency)

        async def _run(f: FieldEntry) -> CheckResult:
            async with semaphore:
                return await check_field(f, store, sections, match_chain)

        results: list[CheckResult] = list(
            await asyncio.gather(*[_run(f) for f in fields])
        )

        n_dev = sum(1 for r in results if r.match is False)
        n_ok  = sum(1 for r in results if r.match is True)
        n_unk = sum(1 for r in results if r.match is None)
        print(f"\nMatch: {n_ok}  |  Deviation: {n_dev}  |  Unmatched: {n_unk}")

        md = render_markdown(args.form, args.legal, results)
        out = Path(args.output)
        out.write_text(md, encoding="utf-8")
        print(f"Report saved to: {out}")


if __name__ == "__main__":
    asyncio.run(main())
