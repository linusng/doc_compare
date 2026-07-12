"""
Context expansion: recursive defined-term / cross-reference retrieval.

Why
---
Legal agreements define terms up front and use them everywhere. A single
retrieved passage is therefore INCOMPLETE on its own:

    "Maturity Date" means the earliest of the Stated Maturity Date and the date
    upon which the Borrowers terminate the Commitments pursuant to Section 3.6.

To actually understand that clause you also need the definition of "Stated
Maturity Date", "Borrowers" and "Commitments", and the text of Section 3.6 —
and whatever THOSE lean on, one level further down.

What this does
--------------
run_context_expansion(index, query) performs an agentic breadth-first search
over the document's definition graph:

  depth 0  seed retrieval for the query (definition / heading / vector search)
  depth 1  extract every referenced term & section from the seed text, retrieve
           each one's definition/section
  depth 2  repeat on the newly retrieved text
  ...      until max_depth (default 2 — at least two stages of coverage) or no
           new references are found.

Reference extraction is DETERMINISTIC (no LLM required, so the loop is fast and
cheap):
  • quoted defined terms:      "Stated Maturity Date"
  • section cross-references:  Section 3.6 / Clause 5.1 / Schedule 2 / Annex A
  • capitalised inline terms:  Borrowers, Commitments — followed ONLY when the
    document actually defines them ('"Borrowers" means …'), which filters out
    ordinary capitalised prose noise.

Every retrieved chunk is returned with the depth it was found at and WHY it was
retrieved (seed / term / section ref), plus a ready-to-use combined context
string for the next pipeline step.

Usage
-----
    from pipeline import DocumentIndex
    from pipeline.context_expansion import run_context_expansion

    index = DocumentIndex.from_pdf("agreement.pdf")
    result = run_context_expansion(index, "Maturity Date")

    for ch in result.chunks:
        print(ch.depth, ch.via, ch.heading)
    next_step_input = result.combined_context
"""

import re

from pydantic import BaseModel

from .models import SectionChunk
from .section_utils import _extract_section_number, _extract_named_prefix
from .find_content import materialize_candidate, retrieve_content_candidates


# ── Tunables ──────────────────────────────────────────────────────────────────

MAX_DEPTH = 2                 # expansion stages beyond the seed (>= 2 per spec)
MAX_TERMS_PER_DEPTH = 25      # cap new references followed per stage
MAX_TOTAL_CHUNKS = 60         # hard cap on returned chunks
SEED_TOP_K = 4                # vector candidates for the seed query


# ── Result models ─────────────────────────────────────────────────────────────

class ExpandedChunk(BaseModel):
    """One retrieved chunk plus how/why it entered the context set."""
    chunk_id: int
    heading: str
    content: str
    pages: list = []
    depth: int = 0            # 0 = seed, 1 = first expansion, ...
    via: str = "seed"         # "seed" | "term: X" | "section: Section 3.6"

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class ContextExpansionResult(BaseModel):
    """
    All chunks gathered by the recursive expansion, in discovery order
    (seeds first, then depth 1, depth 2, …).

    terms_followed : references actually retrieved, per depth.
    terms_dangling : references seen but not found in the document.
    combined_context : formatted string of every chunk, for the next step.
    """
    query: str
    chunks: list[ExpandedChunk] = []
    terms_followed: dict[int, list[str]] = {}
    terms_dangling: list[str] = []

    @property
    def combined_context(self) -> str:
        parts = []
        for ch in self.chunks:
            parts.append(
                f"SECTION : {ch.heading}  (pages {ch.pages}; depth {ch.depth}; "
                f"via {ch.via})\n{'─' * 50}\n{ch.full_text}"
            )
        return "\n\n".join(parts) if parts else "(no context retrieved)"

    @property
    def headings(self) -> list[str]:
        return [ch.heading for ch in self.chunks]


# ── Reference extraction (deterministic) ──────────────────────────────────────

# Quoted defined terms: "Stated Maturity Date" / “Commitments”.
_QUOTED_TERM_RE = re.compile(r'["“]([A-Z][A-Za-z0-9&\-\' ]{2,60}?)["”]')

# Section cross-references: Section 3.6, Clause 5.1(a), Article IV, § 2.3,
# Schedule 2, Annex A, Appendix 1, Exhibit B, Part III.
_SECTION_REF_RE = re.compile(
    r"\b("
    r"(?:Section|Clause|Article|Paragraph)s?\s+(?:\d+(?:\.\d+)*|[IVXLCDM]+\b)"
    r"|§\s*\d+(?:\.\d+)*"
    r"|(?:Schedule|Annex|Appendix|Exhibit|Part)\s+(?:\d+|[A-Z]\b|[IVXLCDM]+\b)"
    r")",
    re.IGNORECASE,
)

# Inline capitalised phrases (defined terms used without quotes): "Borrowers",
# "Stated Maturity Date". ALL-CAPS tokens like EBITDA are included. Continuation
# is same-line only ([ \t], not \n) so a heading and the next line's sentence
# start are never glued into one phantom term.
_CAP_PHRASE_RE = re.compile(
    r"\b((?:[A-Z][a-z][A-Za-z0-9\-']*|[A-Z]{2,})"
    r"(?:[ \t]+(?:[A-Z][a-z][A-Za-z0-9\-']*|[A-Z]{2,}))*)\b"
)

# Words that start ordinary sentences / structural labels, never defined terms.
# NOTE: 'Date' is deliberately NOT a stopword — countless defined terms end in
# it ("Stated Maturity Date").
_TERM_STOPWORDS = {
    "the", "this", "that", "these", "those", "a", "an", "if", "in", "on", "of",
    "no", "any", "each", "all", "such", "subject", "notwithstanding", "provided",
    "section", "clause", "article", "paragraph", "schedule", "annex", "appendix",
    "exhibit", "part", "page", "means", "shall", "may", "definitions",
}


def _norm_term(term: str) -> str:
    return " ".join((term or "").split()).lower()


def extract_references(text: str) -> tuple[list[str], list[str]]:
    """
    Pull (terms, section_refs) out of a passage, deterministically.

    terms        : quoted defined terms + capitalised inline phrases (deduped,
                   document-order; quoted ones first since they are certain).
    section_refs : normalised cross-references ("Section 3.6", "Schedule 2").
    """
    terms: list[str] = []
    seen: set[str] = set()

    def _add(t: str) -> None:
        t = t.strip()
        n = _norm_term(t)
        if len(n) < 3 or n in _TERM_STOPWORDS or n in seen:
            return
        seen.add(n)
        terms.append(t)

    for m in _QUOTED_TERM_RE.finditer(text):
        _add(m.group(1))

    # Strip section refs before scanning capitalised phrases so "Section 3.6"
    # does not surface "Section" as a term.
    stripped = _SECTION_REF_RE.sub(" ", text)
    for m in _CAP_PHRASE_RE.finditer(stripped):
        phrase = m.group(1)
        words = phrase.split()
        # Trim leading/trailing stopwords picked up at sentence starts.
        while words and words[0].lower() in _TERM_STOPWORDS:
            words = words[1:]
        while words and words[-1].lower() in _TERM_STOPWORDS:
            words = words[:-1]
        if words:
            _add(" ".join(words))

    refs: list[str] = []
    seen_refs: set[str] = set()
    for m in _SECTION_REF_RE.finditer(text):
        ref = " ".join(m.group(1).split())
        key = _norm_term(ref)
        if key not in seen_refs:
            seen_refs.add(key)
            refs.append(ref)
    return terms, refs


# ── Deterministic lookups in the index ────────────────────────────────────────

def _definition_re(term: str) -> re.Pattern:
    """Regex matching a definition of `term`: '"Term" means / shall mean /
    has the meaning', with quotes optional and case-insensitive."""
    return re.compile(
        r'["“”\']?' + re.escape(term.strip()) +
        r'["“”\']?\s+(?:means|shall\s+mean|has\s+the\s+meaning)',
        re.IGNORECASE,
    )


def _term_variants(term: str) -> list[str]:
    """The term plus singular/plural variants ('Borrowers' ↔ 'Borrower')."""
    t = term.strip()
    variants = [t]
    if t.lower().endswith("ies"):
        variants.append(t[:-3] + "y")
    elif t.lower().endswith("s") and len(t) > 3:
        variants.append(t[:-1])
    else:
        variants.append(t + "s")
    return variants


def find_definition_chunks(term: str, chunks: list[SectionChunk]) -> list[SectionChunk]:
    """Chunks that DEFINE `term` (or its singular/plural variant)."""
    for variant in _term_variants(term):
        pattern = _definition_re(variant)
        hits = [c for c in chunks if pattern.search(c.full_text)]
        if hits:
            return hits
    return []


def has_definition(term: str, chunks: list[SectionChunk]) -> bool:
    return bool(find_definition_chunks(term, chunks))


def _ref_number(ref: str) -> str | None:
    """'Section 3.6' → '3.6'; 'Article IV' → '4'; 'Schedule 2' → None (named)."""
    return _extract_section_number(ref)


def find_section_chunks(ref: str, chunks: list[SectionChunk]) -> list[SectionChunk]:
    """
    Chunks belonging to a cross-referenced section: exact section number match
    (all split parts) for numbered refs; heading-prefix match for named refs
    ('Schedule 2', 'Annex A').
    """
    num = _ref_number(ref)
    if num:
        return [c for c in chunks if _extract_section_number(c.heading) == num]

    ref_n = _norm_term(ref)
    hits = []
    for c in chunks:
        head_n = _norm_term(c.heading)
        if head_n.startswith(ref_n) or _norm_term(_extract_named_prefix(c.heading) or "") == ref_n:
            hits.append(c)
    return hits


def find_heading_chunks(term: str, chunks: list[SectionChunk]) -> list[SectionChunk]:
    """Chunks whose heading starts with the term (e.g. a '5. Repayment' heading
    for the term 'Repayment')."""
    t = _norm_term(term)
    return [c for c in chunks if _norm_term(c.heading).lstrip("0123456789. ").startswith(t)]


# ── Seed retrieval ────────────────────────────────────────────────────────────

def _seed_chunks(index, query: str, top_k: int = SEED_TOP_K) -> list[SectionChunk]:
    """
    First-stage retrieval for the query: definition match, then heading match,
    then vector search (when the index has a vector store). Deterministic paths
    first so a defined-term query lands exactly on its definition.
    """
    hits = find_definition_chunks(query, index.chunks)
    if hits:
        return hits

    hits = find_heading_chunks(query, index.chunks)
    if hits:
        return hits

    store = getattr(index, "vector_store", None)
    if store is None:
        return []
    out: list[SectionChunk] = []
    by_id = {c.chunk_id: c for c in index.chunks}
    for doc, _score in retrieve_content_candidates(index, query, top_k=top_k):
        usable = materialize_candidate(doc, index)
        if usable is None:
            continue
        cid = usable.metadata.get("chunk_id")
        if cid in by_id:
            out.append(by_id[cid])
    return out


# ── Main entry point ──────────────────────────────────────────────────────────

def run_context_expansion(
    index,
    query: str,
    max_depth: int = MAX_DEPTH,
    max_terms_per_depth: int = MAX_TERMS_PER_DEPTH,
    max_total_chunks: int = MAX_TOTAL_CHUNKS,
    verbose: bool = True,
) -> ContextExpansionResult:
    """
    Recursively retrieve everything the answer to `query` depends on.

    BFS over the agreement's definition graph: seed retrieval, then for each
    stage extract the defined terms and section cross-references mentioned in the
    newly retrieved text and retrieve THEIR definitions/sections, down to
    `max_depth` stages (default 2). Capitalised inline terms are only followed
    when the document actually defines them, so prose noise is not chased.

    Returns every chunk gathered, tagged with depth and reason, plus a
    combined_context string for the next pipeline step.
    """
    if verbose:
        print(f"\n[expand] Context expansion for: '{query}' (max_depth={max_depth})")

    result = ContextExpansionResult(query=query)
    seen_chunks: set[int] = set()
    seen_terms: set[str] = {_norm_term(query)}

    def _add_chunk(chunk: SectionChunk, depth: int, via: str) -> bool:
        if chunk.chunk_id in seen_chunks or len(result.chunks) >= max_total_chunks:
            return False
        seen_chunks.add(chunk.chunk_id)
        result.chunks.append(ExpandedChunk(
            chunk_id=chunk.chunk_id, heading=chunk.heading, content=chunk.content,
            pages=chunk.pages, depth=depth, via=via,
        ))
        return True

    # ── depth 0: seed ─────────────────────────────────────────────────────────
    seeds = _seed_chunks(index, query)
    for c in seeds:
        _add_chunk(c, 0, "seed")
    if verbose:
        print(f"      → depth 0: {len(seeds)} seed chunk(s): "
              f"{[c.heading[:50] for c in seeds[:4]]}")
    if not seeds:
        return result

    # ── depths 1..max_depth: follow references from the previous stage ───────
    frontier = list(result.chunks)          # chunks added at the previous depth
    for depth in range(1, max_depth + 1):
        # Collect references mentioned in the frontier text.
        wanted: list[tuple[str, str]] = []  # (kind, value); kind in {term, section}
        for ch in frontier:
            # Extract from the BODY only: heading text would leak its own words
            # ("Definitions", "(part 2)") as phantom terms.
            terms, refs = extract_references(ch.content)
            for r in refs:
                if _norm_term(r) not in seen_terms:
                    wanted.append(("section", r))
                    seen_terms.add(_norm_term(r))
            for t in terms:
                if _norm_term(t) in seen_terms:
                    continue
                # Follow unquoted capitalised phrases only if the doc defines them.
                if has_definition(t, index.chunks) or find_heading_chunks(t, index.chunks):
                    wanted.append(("term", t))
                    seen_terms.add(_norm_term(t))
                else:
                    seen_terms.add(_norm_term(t))       # don't re-examine
                    result.terms_dangling.append(t)

        wanted = wanted[:max_terms_per_depth]
        if not wanted:
            if verbose:
                print(f"      → depth {depth}: no new references — stopping early")
            break

        followed: list[str] = []
        new_frontier: list[ExpandedChunk] = []
        for kind, value in wanted:
            if kind == "section":
                hits = find_section_chunks(value, index.chunks)
                via = f"section: {value}"
            else:
                hits = find_definition_chunks(value, index.chunks) \
                       or find_heading_chunks(value, index.chunks)
                via = f"term: {value}"
            if not hits:
                result.terms_dangling.append(value)
                continue
            followed.append(value)
            for c in hits:
                if _add_chunk(c, depth, via):
                    new_frontier.append(result.chunks[-1])

        result.terms_followed[depth] = followed
        if verbose:
            print(f"      → depth {depth}: followed {len(followed)} reference(s) "
                  f"→ +{len(new_frontier)} chunk(s): {followed[:8]}")
        if not new_frontier:
            break
        frontier = new_frontier

    if verbose:
        print(f"      → TOTAL: {len(result.chunks)} chunk(s); "
              f"dangling refs: {len(result.terms_dangling)}")
    return result


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from .ingestion import DocumentIndex

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline.context_expansion <pdf_path> <query>")
        raise SystemExit(1)

    idx = DocumentIndex.from_pdf(sys.argv[1])
    res = run_context_expansion(idx, sys.argv[2])
    print("\n" + "=" * 60)
    for ch in res.chunks:
        print(f"  d{ch.depth} [{ch.via[:40]:<40}] {ch.heading[:60]}")
    print(f"\nDangling (not found in doc): {res.terms_dangling}")
