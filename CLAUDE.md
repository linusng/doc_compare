# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Tool

```bash
# Install dependencies
pip install pymupdf python-docx openai pydantic tiktoken

# Set API key
export OPENAI_API_KEY="your-key"

# Basic usage
python doc_comparison.py --base base.pdf --compare offer.docx

# With custom endpoint and model
python doc_comparison.py --base base.pdf --compare offer.docx \
    --output report.md \
    --base-url https://your-endpoint/v1 \
    --model gpt-oss-120b \
    --context-limit 131072
```

## Architecture

Single-file CLI tool (`doc_comparison.py`) that compares legal documents (PDF vs DOCX) using a full-context LLM call — no RAG, no chunking.

### Pipeline

1. **Document Loading** — `load_pdf` (PyMuPDF) and `load_docx` (python-docx) extract text into `PagedDocument` with per-paragraph page numbers. DOCX page detection uses `w:lastRenderedPageBreak` markers.

2. **Section Inference** — `infer_sections` heuristically tags each paragraph with its nearest heading (title case, all caps, or numbered), giving the LLM clause context.

3. **LLM Comparison** — Both full documents are sent in a single prompt. A token guard (tiktoken, 85% safety margin) pre-validates that both docs fit within the context window. Response is forced to JSON via `response_format`.

4. **Report Generation** — `render_markdown` formats deviations into a markdown table with source page numbers, clause text, and LLM commentary.

### Key Data Models (Pydantic v2)

- `Paragraph` — text, page (1-based), doc_name, section
- `PagedDocument` — doc_name + list of Paragraphs
- `DeviationItem` — item_no, section, base_page, compare_page, base/compare paragraph, deviation flag, comments (required when `deviation=True`, enforced by validator)
- `ComparisonResult` / `ComparisonReport` — top-level output wrappers

### Design Rationale

- **No RAG**: Both documents fit in 128K context; chunking would destroy cross-clause context and embedding similarity is poor for legal materiality
- **Temperature 0**: Deterministic output for audit reliability
- **OpenAI-compatible client**: Works with OpenAI or any compatible endpoint via `--base-url`
