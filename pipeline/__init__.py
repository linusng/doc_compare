"""
pipeline — shared document processing infrastructure.

Quick start
-----------
Build the index once per PDF, then query it as many times as needed:

    from pipeline import DocumentIndex
    from pipeline.extract import run_extract
    from pipeline.collate import run_collate

    index = DocumentIndex.from_pdf("agreement.pdf")

    result = run_extract(index, "Schedule 2 Compounded Rate Terms")
    result = run_extract(index, "3.1 Purpose", verify=True)
    result = run_collate(index, "Extension Option: the Final Maturity Date...")

Public API
----------
DocumentIndex       Build once, reuse many times.
ExtractionResult    Output of run_extract().
CollatedResult      Output of run_collate().
SectionEvidence     One section's evidence within a CollatedResult.
DeviationAnalysis   Structured deviation verdict from cp_checker.
"""

from .ingestion import DocumentIndex
from .models import CollatedResult, ExtractionResult, SectionEvidence, SectionChunk
from .cp_checker import DeviationAnalysis

__all__ = [
    "DocumentIndex",
    "ExtractionResult",
    "CollatedResult",
    "SectionEvidence",
    "SectionChunk",
    "DeviationAnalysis",
]
