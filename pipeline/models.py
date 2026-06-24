"""
Shared Pydantic data models used across the pipeline.
"""
import pydantic
from pydantic import BaseModel


def pydantic_copy(model, update: dict):
    """Pydantic v1/v2 compatible model copy with field overrides."""
    if int(pydantic.VERSION.split(".")[0]) >= 2:
        return model.model_copy(update=update)
    return model.copy(update=update)


class TextBlock(BaseModel):
    """A single text block extracted from a PDF page."""
    text: str
    page: int
    bbox: tuple
    font_size: float
    is_bold: bool
    block_no: int
    is_heading: bool = False
    page_height: float = 0.0   # height of the source page, for margin detection


class SectionChunk(BaseModel):
    """
    One document section: a heading and all body text beneath it.
    Oversized sections are split into (part 1), (part 2), … chunks
    that share the same base heading.
    """
    heading: str
    content: str
    pages: list
    start_page: int
    chunk_id: int = 0
    heading_level: int = 0  # 0=unknown, 1=top-level "1.", 2=sub "1.1", etc.

    @property
    def full_text(self) -> str:
        return f"{self.heading}\n{self.content}".strip()


class ExtractionResult(BaseModel):
    """Result of extract mode: a single merged section."""
    heading: str
    content: str
    pages: list
    chunk_id: int
    verified: bool = False
    llm_output: str = None


class ContentMatch(BaseModel):
    """One confirmed passage matched by run_find."""
    content: str
    heading: str | None = None
    pages: list = []
    chunk_id: int | None = None
    score: float | None = None
    llm_output: str | None = None
    expanded: bool = False   # True if expanded from a header-only section container


class ContentMatchResult(BaseModel):
    """
    Result of find mode (run_find): the passage(s) of content that match a
    free-text query.

    run_find evaluates every candidate and keeps all that the LLM confirms.
    `matches` holds every confirmed passage, ranked best (highest score) first.
    The top-level convenience fields (content, heading, pages, …) mirror the
    single best match, so `result.content` still works.

    `found` is False — `matches` is empty and `content` is None — when no
    candidate is confirmed (the "return No" case). Check `result.found` or read
    `result.answer` ("Yes"/"No").
    """
    query: str
    found: bool = False
    # Best confirmed match (mirrors matches[0]) — convenience for single-result use.
    content: str | None = None
    heading: str | None = None
    pages: list = []
    chunk_id: int | None = None
    score: float | None = None
    llm_output: str | None = None
    # Every confirmed match, ranked best-first.
    matches: list[ContentMatch] = []

    @property
    def answer(self) -> str:
        return "Yes" if self.found else "No"

    @property
    def contents(self) -> list[str]:
        """All confirmed match passages as a plain list of strings, best-first."""
        return [m.content for m in self.matches]


class SectionEvidence(BaseModel):
    """Evidence found in one section relevant to a collation statement."""
    heading: str
    pages: list
    matched_terms: list    # terms from the statement that matched this section
    excerpt: str           # the section text


class CollatedResult(BaseModel):
    """Full collation result across all matched sections."""
    statement: str
    terms_extracted: list       # key terms the LLM pulled from the statement
    match_count: int            # total unique chunks found
    sections_referenced: list   # unique headings in page order
    evidence: list              # list[SectionEvidence]
    combined_context: str       # ready-to-use context string for downstream LLMs
    synthesis: str = ""         # optional LLM synthesis (if synthesize=True)
