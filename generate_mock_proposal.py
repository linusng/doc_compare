"""
Mock Credit Proposal Generator
================================
Generates a realistic credit proposal form PDF with a Field | Value table
structure, matching the facility agreement in files/mock_document.pdf.

Contains 2 deliberate deviations and 1 unmatched field for testing:
  - Prepayment Fee:  proposal says 1.50%/0.75%  (legal says 1.00%/0.50%)
  - DSCR Covenant:   proposal says minimum 1.20x (legal says 1.25x)
  - Governing Law:   not present as a section in the legal document

Usage:
    python generate_mock_proposal.py
    python generate_mock_proposal.py --output files/my_proposal.pdf
"""

import argparse
from pathlib import Path

from fpdf import FPDF, XPos, YPos


# ─────────────────────────────────────────────
# Layout constants  (mm, fpdf2 default unit)
# ─────────────────────────────────────────────

COL1 = 70   # field name column  (mm)
COL2 = 110  # field value column (mm)
ROW  = 8    # row height         (mm)

NL = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}   # replaces ln=1


# ─────────────────────────────────────────────
# PDF class
# ─────────────────────────────────────────────

class ProposalPDF(FPDF):

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 8, "ABC BANK LIMITED", align="C", **NL)
        self.set_font("Helvetica", "B", 12)
        self.cell(0, 7, "CREDIT APPROVAL PROPOSAL FORM", align="C", **NL)
        self.set_font("Helvetica", "", 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, "CONFIDENTIAL - For Internal Use Only", align="C", **NL)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    def section_band(self, title: str) -> None:
        """Full-width coloured band -- NOT a table cell (no border)."""
        self.ln(3)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(COL1 + COL2, ROW, f"  {title}", border=0, fill=True, **NL)
        self.set_text_color(0, 0, 0)
        self.set_font("Helvetica", "", 10)

    def field_row(self, name: str, value: str, shade: bool = False) -> None:
        """Bordered Field | Value row -- detected as table by PyMuPDF."""
        if shade:
            self.set_fill_color(245, 246, 250)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(COL1, ROW, f"  {name}", border=1, fill=shade)
        self.set_font("Helvetica", "", 10)
        self.cell(COL2, ROW, f"  {value}", border=1, fill=shade, **NL)

    def deviation_row(self, name: str, value: str, shade: bool = False) -> None:
        """Same layout but value cell highlighted in amber to flag a deviation."""
        if shade:
            self.set_fill_color(245, 246, 250)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_font("Helvetica", "B", 10)
        self.cell(COL1, ROW, f"  {name}", border=1, fill=shade)
        self.set_fill_color(255, 243, 205)   # amber
        self.set_font("Helvetica", "", 10)
        self.cell(COL2, ROW, f"  {value}", border=1, fill=True, **NL)
        self.set_fill_color(255, 255, 255)


# ─────────────────────────────────────────────
# Document content
# ─────────────────────────────────────────────

def build(pdf: ProposalPDF) -> None:

    pdf.add_page()

    # ── SECTION A: BORROWER DETAILS ───────────────────────────────────────
    pdf.section_band("SECTION A:  BORROWER DETAILS")
    for i, (name, value) in enumerate([
        ("Borrower Name",         "XYZ Corporation Pte. Ltd."),
        ("Registration No.",      "201056789B"),
        ("Registered Address",    "100 Orchard Road, #20-01, Singapore 238840"),
        ("Business Activity",     "Commercial property development"),
        ("Contact Person",        "Mr. Tan Wei Ming, CFO"),
    ]):
        pdf.field_row(name, value, shade=(i % 2 == 0))

    # ── SECTION B: FACILITY DETAILS ──────────────────────────────────────
    pdf.section_band("SECTION B:  FACILITY DETAILS")
    for i, (name, value) in enumerate([
        ("Facility Type",         "Term Loan"),
        ("Facility Amount",       "SGD 50,000,000"),
        ("Currency",              "Singapore Dollars (SGD)"),
        ("Purpose",               "Property Acquisition and Development"),
        ("Property Address",      "100 Industrial Park Road, Singapore 123456"),
        ("Availability Period",   "12 months from agreement date"),
        ("Loan Tenor",            "5 years from first drawdown"),
        ("Drawdown",              "Min SGD 1,000,000; multiples of SGD 500,000"),
    ]):
        pdf.field_row(name, value, shade=(i % 2 == 0))

    # ── SECTION C: PRICING & FEES ─────────────────────────────────────────
    pdf.section_band("SECTION C:  PRICING AND FEES")
    rows_c = [
        ("Interest Rate",         "SORA + 2.50% per annum",                          False),
        ("Interest Basis",        "Actual / 365 days",                               False),
        ("Interest Payment",      "Quarterly in arrear",                             False),
        ("Arrangement Fee",       "1.00% of facility amount (non-refundable)",       False),
        ("Commitment Fee",        "0.50% p.a. on undrawn and uncancelled balance",   False),
        # DELIBERATE DEVIATION -- legal document says 1.00% within 2 yrs, 0.50% thereafter
        ("Prepayment Fee",        "1.50% within 24 months; 0.75% thereafter",        True),
    ]
    for i, (name, value, is_dev) in enumerate(rows_c):
        (pdf.deviation_row if is_dev else pdf.field_row)(name, value, shade=(i % 2 == 0))

    # ── SECTION D: FINANCIAL COVENANTS ───────────────────────────────────
    pdf.section_band("SECTION D:  FINANCIAL COVENANTS")
    rows_d = [
        # DELIBERATE DEVIATION -- legal document says minimum 1.25x
        ("DSCR Covenant",         "Minimum 1.20x (semi-annual testing)",              True),
        ("LTV Ratio",             "Maximum 65% of market value",                      False),
        ("ICR Covenant",          "Minimum 2.00x (semi-annual testing)",              False),
        ("Covenant Testing",      "Semi-annual, audited or management accounts",      False),
    ]
    for i, (name, value, is_dev) in enumerate(rows_d):
        (pdf.deviation_row if is_dev else pdf.field_row)(name, value, shade=(i % 2 == 0))

    # ── SECTION E: REPAYMENT ─────────────────────────────────────────────
    pdf.section_band("SECTION E:  REPAYMENT")
    for i, (name, value) in enumerate([
        ("Repayment Structure",   "16 equal quarterly instalments"),
        ("Instalment Amount",     "SGD 3,125,000 per quarter"),
        ("First Instalment",      "12 months after first drawdown"),
        ("Final Maturity Date",   "60 months (5 years) from first drawdown"),
        ("Default Interest",      "2.00% per annum above prevailing rate"),
    ]):
        pdf.field_row(name, value, shade=(i % 2 == 0))

    # ── SECTION F: OTHER TERMS ───────────────────────────────────────────
    pdf.section_band("SECTION F:  OTHER TERMS")
    for i, (name, value) in enumerate([
        ("Security",              "First legal mortgage over the property"),
        ("Events of Default",     "As per facility agreement standard terms"),
        # UNMATCHED -- no dedicated governing law section in the legal document
        ("Governing Law",         "Laws of Singapore"),
        ("Dispute Resolution",    "Singapore courts (non-exclusive jurisdiction)"),
    ]):
        pdf.field_row(name, value, shade=(i % 2 == 0))

    # ── APPROVAL SIGNATURES ──────────────────────────────────────────────
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "APPROVAL", **NL)
    pdf.set_font("Helvetica", "", 10)

    sig_col = 55
    gap     = 8
    pdf.cell(sig_col, 16, "", border=1)
    pdf.cell(gap,     16, "")
    pdf.cell(sig_col, 16, "", border=1)
    pdf.cell(gap,     16, "")
    pdf.cell(sig_col, 16, "", border=1, **NL)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(sig_col, 5, "Prepared by / Date", align="C")
    pdf.cell(gap,     5, "")
    pdf.cell(sig_col, 5, "Reviewed by / Date", align="C")
    pdf.cell(gap,     5, "")
    pdf.cell(sig_col, 5, "Approved by / Date", align="C", **NL)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def generate(output_path: str) -> None:
    pdf = ProposalPDF(orientation="P", unit="mm", format="A4")
    pdf.set_margins(15, 20, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    build(pdf)
    pdf.output(output_path)
    print(f"Generated: {output_path}")
    print("  Deliberate deviations (highlighted amber in the PDF):")
    print("    - Prepayment Fee:  1.50%/0.75%  (legal: 1.00%/0.50%)")
    print("    - DSCR Covenant:   min 1.20x     (legal: min 1.25x)")
    print("  Unmatched field (no section in legal doc):")
    print("    - Governing Law")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock credit proposal PDF.")
    parser.add_argument(
        "--output", "-o",
        default="files/mock_proposal.pdf",
        help="Output PDF path (default: files/mock_proposal.pdf)",
    )
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate(args.output)


if __name__ == "__main__":
    main()
