"""
Smart indexing (v2): agentic, LLM-driven section parsing.

Why a v2
--------
ingestion.py builds sections from PyMuPDF *blocks* and decides boundaries with
font/bold/number heuristics. That fragments meaning: a parties block such as

    <Company_Name>
    is the Initial Borrower

arrives as two separate blocks, and a bold company name can be mis-promoted to a
"heading", splitting the name from its role. The heuristics have no understanding
that those two lines are ONE thought, so a query like "Who is the Initial
Borrower?" can't retrieve them together.

Approach
--------
Mirror the "smart query" design (agents + tools + a deterministic guard) on the
INDEXING side:

  1. Extract the document as reading-order LINES (page text, not blocks), so
     context is preserved and running headers/footers are dropped.
  2. An LLM agent walks the lines window-by-window and, via an ``emit_section``
     TOOL, marks where each real section begins — understanding that a party's
     name and its role belong together, that the preamble/parties/recitals are
     ONE section, and that '(a)'/'(i)' list items are not headings.
  3. Sections are reconstructed DETERMINISTICALLY by slicing the original lines
     at those boundaries — the LLM never rewrites text, so nothing is invented or
     lost (the same guarantee the query grounding guard gives).

If the LLM/tool backend is unavailable the module falls back to the heuristic
chunker from ingestion.py, so v2 always yields a usable index.

Drop-in
-------
    from pipeline.ingestion_v2 import DocumentIndexV2

    index = DocumentIndexV2.from_pdf("agreement.pdf")   # same shape as DocumentIndex
    run_query(index, "Who is the Initial Borrower?")    # works unchanged

DocumentIndexV2 IS a DocumentIndex (same fields: pdf_path, chunks, vector_store),
so every downstream module (extract/collate/query/find_content) works as-is.
"""

import json
import re
import traceback
from dataclasses import dataclass

import pymupdf
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from .models import SectionChunk, pydantic_copy
from .ingestion import (
    DocumentIndex,
    BGE_M3_MAX_TOKENS,
    TARGET_CHUNK_TOKENS,
    _STRUCT_HEADING_RE,
    _get_heading_level,
    build_vector_store,
    split_oversized_chunks,
    token_len,
)


# ── Tunables ──────────────────────────────────────────────────────────────────

# A window is the slice of lines shown to the LLM in ONE segmentation call. Bigger
# windows mean fewer LLM calls (lower RPM, faster) — the cost is a longer prompt
# per call, which modern models handle easily. These are the primary speed knobs;
# raise WINDOW_TARGET_TOKENS to cut the request count further (bounded by the
# model's usable context). Tune via DocumentIndexV2.from_pdf(window_target_tokens=).
WINDOW_TARGET_TOKENS = 4000
WINDOW_MAX_LINES = 400

# Running header/footer suppression: a short line whose text repeats on at least
# this many pages is treated as a running artefact (page number, running title).
_REPEAT_MIN_PAGES = 4
_REPEAT_MAX_LEN = 120

# A first line longer than this that begins with a structural marker is treated
# as a run-in heading ("Section 2.01 Commitment.  Subject to …") and split so the
# body still lands in the section content.
_RUNIN_MIN_CHARS = 120
_RUNIN_MAX_HEADING_CHARS = 140


# ── Line model ────────────────────────────────────────────────────────────────

@dataclass
class Line:
    """One reading-order line of the document. `index` is its position in the
    final (filtered) line list — the id the LLM refers to when marking sections."""
    index: int
    page: int
    text: str


def extract_page_lines(pdf_path: str) -> list[Line]:
    """
    Extract the document as reading-order lines with page numbers.

    Uses page-level text (not blocks) so a paragraph's lines stay adjacent and in
    order. Blank lines are dropped; running headers/footers are suppressed. The
    surviving lines are re-indexed 0..N-1 so `Line.index` is a stable id.
    """
    doc = pymupdf.open(pdf_path)
    raw: list[Line] = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        for ln in text.split("\n"):
            ln = ln.rstrip()
            if ln.strip():
                raw.append(Line(index=-1, page=page_num, text=ln))
    doc.close()

    kept = _drop_repeating_lines(raw)
    # Re-index sequentially so the LLM's line ids map to list positions.
    return [Line(index=i, page=ln.page, text=ln.text) for i, ln in enumerate(kept)]


def _drop_repeating_lines(lines: list[Line]) -> list[Line]:
    """Drop short lines whose text repeats across many pages (page numbers,
    running titles, footers) — the reading-order analogue of ingestion.py's
    repeating-block filter."""
    from collections import Counter

    seen_on_pages: dict[str, set[int]] = {}
    for ln in lines:
        norm = " ".join(ln.text.split())
        if norm and len(norm) <= _REPEAT_MAX_LEN:
            seen_on_pages.setdefault(norm, set()).add(ln.page)

    repeating = {t for t, pages in seen_on_pages.items() if len(pages) >= _REPEAT_MIN_PAGES}
    if repeating:
        print(f"      → Suppressed {len(repeating)} repeating header/footer line(s)")
    return [ln for ln in lines if " ".join(ln.text.split()) not in repeating]


# ── Windowing ─────────────────────────────────────────────────────────────────

def _format_window(window: list[Line]) -> str:
    """Render a window as numbered, page-tagged lines for the LLM."""
    return "\n".join(f"[{ln.index}] (p{ln.page}) {ln.text}" for ln in window)


def _iter_windows(
    lines: list[Line],
    target_tokens: int = WINDOW_TARGET_TOKENS,
    max_lines: int = WINDOW_MAX_LINES,
):
    """Yield contiguous windows of lines, cut on a line boundary once the token
    budget or line cap is reached. Larger target_tokens → fewer windows → fewer
    LLM calls."""
    start = 0
    n = len(lines)
    while start < n:
        end = start
        while end < n and (end - start) < max_lines:
            trial = lines[start:end + 1]
            if end > start and token_len(_format_window(trial)) > target_tokens:
                break
            end += 1
        yield lines[start:max(end, start + 1)]
        start = max(end, start + 1)


# ── LLM segmentation (agentic, tool-driven) ───────────────────────────────────

_SEG_SYSTEM = (
    "You are a legal-document structure parser. You are given a slice of "
    "consecutive, NUMBERED lines from an agreement — each line is labelled with "
    "its global line id and PDF page: '[id] (pN) text'.\n\n"
    "Identify every line that BEGINS a real section and record it by calling the "
    "`emit_section` tool ONCE per section.\n\n"
    "A section start is:\n"
    "- A numbered clause/article: '1.', '1.1', '2.01', 'ARTICLE II', "
    "'Section 3.1', '§ 4.2'.\n"
    "- A named part: 'SCHEDULE 2', 'ANNEX A', 'APPENDIX 1', 'PART III'.\n"
    "- A defined-term heading, where the document lists definitions.\n"
    "- The opening PREAMBLE. If this slice contains the very start of the "
    "document, emit ONE section at its first line with heading "
    "'Preamble and Parties' covering the title, the parties, and the recitals.\n\n"
    "CRITICAL rules:\n"
    "- NEVER start a section in the middle of a sentence.\n"
    "- NEVER split a party's name from its role/description: e.g. a line with a "
    "company name followed by a line 'is the Initial Borrower' belong to the SAME "
    "(parties/preamble) section — do not put a boundary between them.\n"
    "- Do NOT emit sections for '(a)', '(i)', bullets, or continuation lines — "
    "those are list items inside a section, not headings.\n"
    "- If the slice merely CONTINUES the CURRENT SECTION named below, emit "
    "nothing for those continuation lines.\n\n"
    "For each section start call: emit_section(heading=<the heading text of that "
    "line>, start_line=<the line id>, level=<1 for a top-level clause / ARTICLE / "
    "SCHEDULE, 2 for x.y, 3 for x.y.z>). After recording every section in this "
    "slice, stop."
)


def _emit_section_tool(recorded: list[dict]) -> StructuredTool:
    def emit_section(heading: str, start_line: int, level: int = 0) -> str:
        """Record that a new document section begins at line `start_line`."""
        recorded.append({
            "heading": (heading or "").strip(),
            "line": int(start_line),
            "level": int(level or 0),
        })
        return f"recorded section at line {start_line}: {heading!r}"

    return StructuredTool.from_function(
        func=emit_section,
        name="emit_section",
        description=(
            "Record the start of one document section: its heading text, the line "
            "id where it begins, and its depth level. Call once per section."
        ),
    )


def _parse_boundaries_json(text: str) -> list[dict]:
    """Fallback: parse a JSON array of {line/start_line, heading, level} objects
    from the model's text, for backends that answer in prose instead of tools."""
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out: list[dict] = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        line = item.get("line", item.get("start_line"))
        if line is None:
            continue
        try:
            out.append({
                "heading": str(item.get("heading", "")).strip(),
                "line": int(line),
                "level": int(item.get("level", 0) or 0),
            })
        except (TypeError, ValueError):
            continue
    return out


def _segment_window(
    window: list[Line],
    prior_heading: str,
    llm_model: str,
    base_url: str,
    api_key: str,
    max_steps: int = 1,
    verbose: bool = True,
    window_no: int = 0,
) -> list[dict]:
    """
    Ask the agent to mark section starts within one window via the emit_section
    tool. Returns boundary dicts whose line id falls inside the window (clamped so
    the model can't reference lines it wasn't shown). Falls back to JSON parsing
    if the backend doesn't call tools.

    Cost: exactly `max_steps` LLM calls per window (default 1). A tool-capable
    model emits ALL its emit_section calls in a single response, so one call is
    enough and there is no extra "finishing" round-trip. Raise max_steps only if
    your model splits tool calls across turns.

    With verbose=True, prints per-window diagnostics: how many tool calls the model
    made, how many boundaries were recorded, and — when nothing was recorded — the
    raw model output, so you can see WHY a window produced no sections.
    """
    span = (f"lines {window[0].index}-{window[-1].index}, "
            f"p{window[0].page}-{window[-1].page}") if window else "empty"

    recorded: list[dict] = []
    tool = _emit_section_tool(recorded)
    llm = ChatOpenAI(model=llm_model, base_url=base_url, api_key=api_key, temperature=0)
    llm_with_tools = llm.bind_tools([tool])

    human = (
        f"CURRENT SECTION: {prior_heading}\n\n"
        f"LINES:\n{_format_window(window)}"
    )
    messages: list = [SystemMessage(content=_SEG_SYSTEM), HumanMessage(content=human)]

    last_text = ""
    total_tool_calls = 0
    for step in range(max_steps):
        ai: AIMessage = llm_with_tools.invoke(messages)
        messages.append(ai)
        if ai.content:
            last_text = ai.content if isinstance(ai.content, str) else str(ai.content)
        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls:
            break
        total_tool_calls += len(tool_calls)
        # Execute the tool calls (side effect: populate `recorded`). We only send
        # results back and re-invoke if the caller allows more steps — otherwise
        # one call already gave us every section in this window.
        for call in tool_calls:
            try:
                obs = tool.invoke(call["args"])
            except Exception as exc:   # noqa: BLE001 — bad args from the model
                obs = f"error: {exc}"
            messages.append(ToolMessage(content=str(obs), tool_call_id=call["id"]))
        if step + 1 >= max_steps:
            break

    used_json_fallback = False
    if not recorded:
        recorded = _parse_boundaries_json(last_text)
        used_json_fallback = bool(recorded)

    valid_ids = {ln.index for ln in window}
    kept = [b for b in recorded if b["line"] in valid_ids]

    if verbose:
        print(f"      → [seg] window {window_no} ({span}): "
              f"tool_calls={total_tool_calls}, recorded={len(recorded)}"
              f"{' via JSON-fallback' if used_json_fallback else ''}, "
              f"kept-in-window={len(kept)}")
        if total_tool_calls == 0 and not recorded:
            # The decisive diagnostic: the model neither called the tool nor
            # produced parseable JSON. Show what it actually said.
            preview = (last_text or "(empty response)").strip().replace("\n", " ")
            print(f"           ↳ model did NOT call emit_section. Raw output: "
                  f"{preview[:300]!r}")

    return kept


def segment_document(
    lines: list[Line],
    llm_model: str = "gemma3-27b-it",
    base_url: str = "http://localhost:11434/v1",
    api_key: str = "ollama",
    verbose: bool = True,
    window_target_tokens: int = WINDOW_TARGET_TOKENS,
    max_steps: int = 1,
) -> list[dict]:
    """
    Walk the whole document window-by-window, carrying the last known section
    heading as context, and collect all section boundaries.

    Makes exactly one window list up front so the LLM-call budget (= number of
    windows × max_steps) is known and printed before any calls are made.
    """
    windows = list(_iter_windows(lines, target_tokens=window_target_tokens))
    print(f"      → {len(windows)} window(s) → up to {len(windows) * max_steps} "
          f"LLM call(s) (window_target_tokens={window_target_tokens}, "
          f"max_steps={max_steps})")

    boundaries: list[dict] = []
    prior_heading = "(start of document)"
    windows_with_tool_calls = 0
    for i, window in enumerate(windows, start=1):
        found = _segment_window(
            window, prior_heading, llm_model=llm_model, base_url=base_url,
            api_key=api_key, verbose=verbose, window_no=i, max_steps=max_steps,
        )
        if found:
            windows_with_tool_calls += 1
            prior_heading = found[-1].get("heading") or prior_heading
        boundaries.extend(found)
    print(f"      → Segmented {len(windows)} window(s) → {len(boundaries)} section "
          f"boundary(ies) from {windows_with_tool_calls} productive window(s)")
    return boundaries


# ── Deterministic reconstruction (verbatim slicing) ───────────────────────────

def _split_runin(first_line: str) -> tuple[str, str]:
    """
    Split a run-in heading line ("Section 2.01 Commitment.  Subject to …") into
    (heading, inline_body) so the body is not stranded in the heading. Returns
    (line, "") when the line is a plain heading.
    """
    stripped = first_line.strip()
    if len(stripped) <= _RUNIN_MIN_CHARS or not _STRUCT_HEADING_RE.match(stripped):
        return stripped, ""
    m = _STRUCT_HEADING_RE.match(stripped)
    rest = stripped[m.end():]
    pm = re.search(r"\.\s", rest)
    if not pm:
        return stripped, ""
    cut = m.end() + pm.end()
    head = stripped[:cut].strip()
    body = stripped[cut:].strip()
    if not body or len(head) > _RUNIN_MAX_HEADING_CHARS:
        return stripped, ""
    return head, body


def _slice_to_chunk(
    lines: list[Line],
    start: int,
    end: int,
    heading: str,
    level: int,
    chunk_id: int,
    heading_is_first_line: bool,
) -> SectionChunk:
    """
    Build one SectionChunk from lines[start:end], verbatim.

    heading_is_first_line=True  → lines[start] is the heading; content is the body
                                  beneath it (run-in body preserved).
    heading_is_first_line=False → synthetic heading (e.g. the preamble); every
                                  line in the slice is content.
    """
    slice_lines = lines[start:end]
    pages = sorted({ln.page for ln in slice_lines}) or [0]

    if heading_is_first_line and slice_lines:
        head_text, inline_body = _split_runin(slice_lines[0].text)
        heading = heading or head_text
        body_lines = [inline_body] if inline_body else []
        body_lines += [ln.text for ln in slice_lines[1:]]
        content = "\n".join(body_lines).strip()
    else:
        content = "\n".join(ln.text for ln in slice_lines).strip()

    return SectionChunk(
        heading=heading.strip() or "(untitled)",
        content=content,
        pages=pages,
        start_page=pages[0],
        chunk_id=chunk_id,
        heading_level=level or _get_heading_level(heading) or 1,
    )


def build_sections(lines: list[Line], boundaries: list[dict]) -> list[SectionChunk]:
    """
    Reconstruct SectionChunks by slicing the original lines at the LLM's section
    boundaries. Lines before the first boundary become a 'Preamble and Parties'
    section. Deterministic and lossless — no LLM text.
    """
    if not lines:
        return []

    by_line: dict[int, dict] = {}
    for b in boundaries:
        li = b["line"]
        if 0 <= li < len(lines) and li not in by_line:
            by_line[li] = b
    starts = sorted(by_line)

    sections: list[SectionChunk] = []

    # Preamble: everything before the first marked section.
    first = starts[0] if starts else len(lines)
    if first > 0:
        sections.append(_slice_to_chunk(
            lines, 0, first, heading="Preamble and Parties", level=1,
            chunk_id=len(sections), heading_is_first_line=False,
        ))

    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(lines)
        b = by_line[s]
        heading = b.get("heading") or lines[s].text.strip()[:150]
        sections.append(_slice_to_chunk(
            lines, s, end, heading=heading, level=b.get("level", 0),
            chunk_id=len(sections), heading_is_first_line=True,
        ))

    return sections


# ── Heuristic fallback (LLM/tools unavailable) ────────────────────────────────

def _heuristic_chunks(
    pdf_path: str,
    llm_model: str,
    base_url: str,
    api_key: str,
) -> list[SectionChunk]:
    """Reuse ingestion.py's block pipeline so v2 always produces a usable index
    even when the segmentation backend is down."""
    from .ingestion import (
        extract_structured_blocks, filter_toc_blocks, filter_repeating_blocks,
        filter_positional_headers_footers, split_structural_headings,
        chunk_by_section, filter_short_chunks,
    )
    blocks = extract_structured_blocks(pdf_path)
    blocks = filter_toc_blocks(blocks, llm_model=llm_model, base_url=base_url, api_key=api_key)
    blocks = filter_repeating_blocks(blocks)
    blocks = filter_positional_headers_footers(blocks)
    blocks = split_structural_headings(blocks)
    chunks = chunk_by_section(blocks)
    return filter_short_chunks(chunks)


# ── DocumentIndexV2: drop-in replacement built with smart indexing ────────────

@dataclass
class DocumentIndexV2(DocumentIndex):
    """
    A DocumentIndex built by the agentic v2 indexer. Same fields/shape as
    DocumentIndex, so it is a drop-in for run_extract / run_collate / run_query /
    run_find.
    """

    @classmethod
    def from_pdf(
        cls,
        pdf_path: str,
        llm_model: str = "gemma3-27b-it",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        embedding_model: str = "bge-m3",
        target_chunk_tokens: int = TARGET_CHUNK_TOKENS,
        verbose: bool = True,
        allow_fallback: bool = True,
        window_target_tokens: int = WINDOW_TARGET_TOKENS,
        segmentation_max_steps: int = 1,
    ) -> "DocumentIndexV2":
        """
        Build a DocumentIndex using LLM-driven section parsing.

        1. Extract reading-order lines (headers/footers dropped).
        2. Agentically segment into sections (with heuristic fallback).
        3. Reconstruct SectionChunks verbatim at the boundaries.
        4. Token-safe split + vector store — identical to v1 from here on.

        verbose        : print per-window segmentation diagnostics (default True).
        allow_fallback : if False, raise instead of silently falling back to the
                         heuristic BLOCK chunker — use this to surface the real
                         segmentation error while debugging.
        window_target_tokens : lines per LLM segmentation call. Larger → fewer
                         windows → fewer LLM calls (lower RPM, faster). Default 4000.
        segmentation_max_steps : LLM calls per window (default 1). One is enough
                         for models that emit all tool calls in a single response.
        """
        print(f"[1/4] Extracting reading-order lines from: {pdf_path}")
        lines = extract_page_lines(pdf_path)
        print(f"      → {len(lines)} lines after header/footer suppression")

        print("[2/4] Agentic section segmentation (LLM + emit_section tool)...")
        sections: list[SectionChunk] = []
        seg_error: Exception | None = None
        try:
            boundaries = segment_document(
                lines, llm_model=llm_model, base_url=base_url, api_key=api_key,
                verbose=verbose, window_target_tokens=window_target_tokens,
                max_steps=segmentation_max_steps,
            )
            sections = build_sections(lines, boundaries)
            print(f"      → LLM segmentation produced {len(sections)} section(s)")
        except Exception as exc:   # noqa: BLE001 — backend/tooling unavailable
            seg_error = exc
            print(f"      → ✗ Segmentation RAISED {type(exc).__name__}: {exc}")
            if verbose:
                traceback.print_exc()

        if len(sections) < 2:
            # Spell out exactly why we are about to use the block-based chunker.
            if seg_error is not None:
                reason = (f"segmentation raised {type(seg_error).__name__} — the "
                          f"LLM/tool backend errored (see traceback above). Common "
                          f"cause: the endpoint/model does not support tool calling.")
            else:
                reason = ("segmentation ran but returned too few boundaries "
                          f"({len(sections)} section(s)) — the model did not call "
                          f"emit_section (see per-window raw output above). Common "
                          f"cause: the model ignores bound tools, or the headings "
                          f"weren't recognised.")
            print(f"      → ⚠  FALLBACK TRIGGERED — reason: {reason}")
            if not allow_fallback:
                raise RuntimeError(
                    f"v2 segmentation failed and allow_fallback=False: {reason}"
                ) from seg_error
            print("      → ⚠  USING HEURISTIC BLOCK CHUNKER (this is the v1 "
                  "block-based path — NOT the smart LLM sections)")
            sections = _heuristic_chunks(pdf_path, llm_model, base_url, api_key)
            print(f"      → Heuristic fallback produced {len(sections)} "
                  f"block-based section(s)")
        else:
            print(f"      → ✓ Using {len(sections)} LLM-parsed section(s) "
                  f"(smart indexing)")

        print("[3/4] Splitting oversized chunks...")
        chunks = split_oversized_chunks(sections, max_tokens=target_chunk_tokens)
        print(f"      → {len(chunks)} chunks after token-safe split "
              f"(target {min(target_chunk_tokens, BGE_M3_MAX_TOKENS)} tokens)")

        print("[4/4] Building vector store...")
        vector_store = build_vector_store(
            chunks, base_url=base_url, api_key=api_key, model=embedding_model,
        )
        print(f"      → Vector store ready ({len(chunks)} documents indexed)")

        return cls(pdf_path=pdf_path, chunks=chunks, vector_store=vector_store)
