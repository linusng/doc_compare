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
