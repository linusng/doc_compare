"""
Mock PDF Generator
==================
Generates a realistic credit facility agreement PDF with hierarchical sections
for testing parse_sections.py.

Section structure:
    Preamble (recitals before any numbered section)
    1.      Definitions and Interpretation
    1.1     Definitions
    1.2     Interpretation
    1.3     Headings
    2.      The Facility
    2.1     Amount and Purpose
    2.2     Availability Period
    2.3     Conditions Precedent
    3.      Interest and Fees
    3.1     Interest Rate
    3.2     Default Interest
    3.3     Financial Covenants          ← parent of 3.3.x
    3.3.1   Debt Service Coverage Ratio
    3.3.2   Loan to Value Ratio
    3.3.3   Interest Coverage Ratio
    3.4     Fees
    4.      Repayment
    4.1     Repayment Schedule
    4.2     Voluntary Prepayment
    5.      Representations and Warranties
    5.1     Status
    5.2     Financial Condition
    6.      Events of Default
    6.1     Non-Payment
    6.2     Breach of Covenant
    6.3     Insolvency

Usage:
    python generate_mock_pdf.py
    python generate_mock_pdf.py --output files/my_mock.pdf
"""

import argparse
from pathlib import Path

import pymupdf


# ─────────────────────────────────────────────
# Page geometry
# ─────────────────────────────────────────────

PAGE_W   = 595   # A4 width  (points)
PAGE_H   = 842   # A4 height (points)
MARGIN_L = 72    # 1 inch left
MARGIN_R = 72    # 1 inch right
MARGIN_T = 80    # top
MARGIN_B = 72    # bottom


# ─────────────────────────────────────────────
# Typography
# ─────────────────────────────────────────────
# PyMuPDF built-in fonts: helv = Helvetica, hebo = Helvetica-Bold

_STYLES: dict[str, dict] = {
    "title":    {"font": "hebo", "size": 16, "before": 0,  "after": 6,  "color": (0, 0, 0)},
    "subtitle": {"font": "helv", "size": 10, "before": 4,  "after": 16, "color": (0.3, 0.3, 0.3)},
    "rule":     {"font": "helv", "size": 1,  "before": 0,  "after": 14, "color": (0, 0, 0)},
    "h1":       {"font": "hebo", "size": 12, "before": 18, "after": 5,  "color": (0, 0, 0)},
    "h2":       {"font": "hebo", "size": 11, "before": 12, "after": 4,  "color": (0, 0, 0)},
    "h3":       {"font": "hebo", "size": 10, "before": 8,  "after": 3,  "color": (0.15, 0.15, 0.15)},
    "body":     {"font": "helv", "size": 10, "before": 4,  "after": 2,  "color": (0, 0, 0)},
    "label":    {"font": "hebo", "size": 9,  "before": 3,  "after": 1,  "color": (0.2, 0.2, 0.2)},
}


# ─────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────

class Renderer:
    def __init__(self) -> None:
        self.doc        = pymupdf.Document()
        self.page       = None
        self.y          = MARGIN_T
        self._page_no   = 0
        self._new_page()

    def _new_page(self) -> None:
        self.page     = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        self._page_no += 1
        self.y         = MARGIN_T
        self._add_header_footer()

    def _add_header_footer(self) -> None:
        """
        Insert a running header and footer on the current page.
        Both sit within the margin strip so parse_sections can filter them out.

        Header strip: y ∈ [15, 35]  (well within top 50 pt)
        Footer strip: y ∈ [812, 828] (well within bottom 50 pt; PAGE_H - 50 = 792)
        """
        # ── Running header ─────────────────────────────────────────────
        header_rect = pymupdf.Rect(MARGIN_L, 15, PAGE_W - MARGIN_R, 35)
        self.page.insert_textbox(
            header_rect,
            "CREDIT FACILITY AGREEMENT",
            fontsize=8,
            fontname="helv",
            color=(0.5, 0.5, 0.5),
            align=1,  # centre
        )
        # Thin rule under the header
        self.page.draw_line(
            pymupdf.Point(MARGIN_L, 37),
            pymupdf.Point(PAGE_W - MARGIN_R, 37),
            color=(0.75, 0.75, 0.75),
            width=0.4,
        )

        # ── Running footer ─────────────────────────────────────────────
        footer_rect = pymupdf.Rect(MARGIN_L, 812, PAGE_W - MARGIN_R, 828)
        self.page.insert_textbox(
            footer_rect,
            f"Confidential  |  ABC Bank Limited  |  Page {self._page_no}",
            fontsize=8,
            fontname="helv",
            color=(0.5, 0.5, 0.5),
            align=1,  # centre
        )

    def _remaining(self) -> float:
        return PAGE_H - MARGIN_B - self.y

    def _write(self, text: str, style: str) -> None:
        cfg = _STYLES[style]
        fontsize  = cfg["size"]
        fontname  = cfg["font"]
        color     = cfg["color"]
        before    = cfg["before"]
        after     = cfg["after"]

        self.y += before

        # Try to render; if overflow, push to a new page and retry once
        for attempt in range(2):
            rect = pymupdf.Rect(MARGIN_L, self.y, PAGE_W - MARGIN_R, PAGE_H - MARGIN_B)
            if rect.height < fontsize * 2.5:
                self._new_page()
                self.y += before
                continue

            result = self.page.insert_textbox(
                rect,
                text,
                fontsize=fontsize,
                fontname=fontname,
                color=color,
                align=0,  # left-align
            )

            if result >= 0:
                # Height used = rect.height − remaining_space
                self.y += (rect.height - result) + after
                return

            # Overflow on attempt 0 → start a new page and retry
            self._new_page()
            self.y += before

    def title(self, text: str)    -> None: self._write(text, "title")
    def subtitle(self, text: str) -> None: self._write(text, "subtitle")
    def h1(self, text: str)       -> None: self._write(text, "h1")
    def h2(self, text: str)       -> None: self._write(text, "h2")
    def h3(self, text: str)       -> None: self._write(text, "h3")
    def body(self, text: str)     -> None: self._write(text, "body")
    def label(self, text: str)    -> None: self._write(text, "label")

    def rule(self) -> None:
        """Draw a thin horizontal rule across the content area."""
        cfg = _STYLES["rule"]
        self.y += cfg["before"]
        self.page.draw_line(
            pymupdf.Point(MARGIN_L, self.y),
            pymupdf.Point(PAGE_W - MARGIN_R, self.y),
            color=(0.6, 0.6, 0.6),
            width=0.5,
        )
        self.y += cfg["after"]

    def save(self, path: str) -> None:
        self.doc.save(path)
        self.doc.close()


# ─────────────────────────────────────────────
# Document content
# ─────────────────────────────────────────────

def build(r: Renderer) -> None:

    # ── Cover / Preamble ──────────────────────────────────────────────────
    r.title("CREDIT FACILITY AGREEMENT")
    r.subtitle("ABC BANK LIMITED  (\"Lender\")\nXYZ CORPORATION PTE. LTD.  (\"Borrower\")")
    r.rule()

    r.body(
        "THIS AGREEMENT is dated 1 April 2026 and is made between ABC Bank Limited, "
        "a company incorporated in Singapore (Registration No. 200012345A), whose "
        "registered office is at 1 Marina Boulevard, Singapore 018989 (the \"Lender\"); "
        "and XYZ Corporation Pte. Ltd., a company incorporated in Singapore "
        "(Registration No. 201056789B), whose registered office is at 100 Orchard Road, "
        "#20-01, Singapore 238840 (the \"Borrower\")."
    )
    r.body(
        "WHEREAS the Borrower has requested the Lender to make available a term loan "
        "facility for the purposes described herein, and the Lender has agreed to do so "
        "on the terms and subject to the conditions set out in this Agreement."
    )
    r.body(
        "IT IS AGREED as follows:"
    )

    # ── Section 1 ─────────────────────────────────────────────────────────
    r.h1("1. DEFINITIONS AND INTERPRETATION")
    r.body(
        "This Clause 1 sets out the defined terms and rules of interpretation that "
        "apply throughout this Agreement. Capitalised terms used but not defined "
        "elsewhere shall have the meanings given to them in this Clause."
    )

    r.h2("1.1 Definitions")
    r.body("In this Agreement, the following terms shall have the meanings set out below:")
    r.body(
        "\"Availability Period\" means the period commencing on the date of this "
        "Agreement and ending on the date falling 12 months thereafter, or such "
        "later date as the Lender may agree in writing."
    )
    r.body(
        "\"Business Day\" means a day (other than a Saturday, Sunday, or gazetted "
        "public holiday) on which commercial banks are open for general business "
        "in Singapore."
    )
    r.body(
        "\"Facility\" means the term loan facility made available under Clause 2, "
        "in an aggregate principal amount not exceeding SGD 50,000,000."
    )
    r.body(
        "\"Interest Period\" means each period of three months, commencing on the "
        "first Drawdown Date and on each subsequent Interest Payment Date thereafter, "
        "and ending on the next succeeding Interest Payment Date."
    )
    r.body(
        "\"Material Adverse Effect\" means any event or circumstance that has or is "
        "reasonably likely to have a material adverse effect on the financial "
        "condition, operations, or assets of the Borrower, or on the ability of the "
        "Borrower to perform its payment obligations under this Agreement."
    )
    r.body(
        "\"SORA\" means the Singapore Overnight Rate Average as published by the "
        "Monetary Authority of Singapore, or any successor rate thereto."
    )

    r.h2("1.2 Interpretation")
    r.body(
        "Unless the context otherwise requires: references to Clauses and Schedules "
        "are to clauses and schedules of this Agreement; words in the singular include "
        "the plural and vice versa; and references to a person include its successors "
        "and permitted assigns."
    )
    r.body(
        "Any reference to a statute or statutory provision includes a reference to that "
        "statute or statutory provision as amended, extended, or re-enacted from time to "
        "time, and to any subordinate legislation made under it."
    )

    r.h2("1.3 Headings")
    r.body(
        "Clause and schedule headings are inserted for convenience only and shall not "
        "affect the construction or interpretation of this Agreement."
    )

    # ── Section 2 ─────────────────────────────────────────────────────────
    r.h1("2. THE FACILITY")
    r.body(
        "Subject to the terms and conditions of this Agreement, the Lender agrees to "
        "make available to the Borrower a term loan facility on the basis set out "
        "in this Clause 2."
    )

    r.h2("2.1 Amount and Purpose")
    r.body(
        "The Facility is a term loan in an aggregate principal amount not exceeding "
        "SGD 50,000,000 (Singapore Dollars Fifty Million only)."
    )
    r.body(
        "The Borrower shall apply the proceeds of each drawdown solely for the purpose "
        "of financing the acquisition and development of the commercial property at "
        "100 Industrial Park Road, Singapore 123456, and for no other purpose."
    )

    r.h2("2.2 Availability Period")
    r.body(
        "The Facility shall be available for drawdown during the Availability Period "
        "in a minimum amount of SGD 1,000,000 per drawdown request and in integral "
        "multiples of SGD 500,000 thereafter."
    )
    r.body(
        "Amounts repaid or prepaid under the Facility may not be re-borrowed. Any "
        "portion of the Facility that remains undrawn at the expiry of the Availability "
        "Period shall be automatically and irrevocably cancelled."
    )

    r.h2("2.3 Conditions Precedent")
    r.body(
        "The Lender's obligation to make available any portion of the Facility is "
        "subject to the Lender having received, in form and substance satisfactory "
        "to it, all of the documents and evidence listed in Schedule 1."
    )
    r.body(
        "Conditions precedent include certified copies of the Borrower's constitutional "
        "documents, evidence of board authorisation, executed security documents, and "
        "such other documents as the Lender may reasonably require prior to drawdown."
    )

    # ── Section 3 ─────────────────────────────────────────────────────────
    r.h1("3. INTEREST AND FEES")
    r.body(
        "This Clause 3 governs the interest rate, financial maintenance covenants, "
        "and fees applicable to the Facility for the duration of this Agreement."
    )

    r.h2("3.1 Interest Rate")
    r.body(
        "Interest shall accrue on the outstanding principal amount of the Facility at "
        "the rate of SORA plus a margin of 2.50% per annum. Interest shall be "
        "calculated on the basis of the actual number of days elapsed in a 365-day year."
    )
    r.body(
        "Interest shall be payable quarterly in arrear on each Interest Payment Date. "
        "The first Interest Payment Date shall fall three calendar months after the "
        "date of the first drawdown under the Facility."
    )

    r.h2("3.2 Default Interest")
    r.body(
        "If the Borrower fails to pay any sum payable by it under this Agreement on "
        "its due date, default interest shall accrue on the overdue amount from the "
        "due date to the date of actual payment at the rate of 2.00% per annum above "
        "the rate that would otherwise be applicable."
    )
    r.body(
        "Default interest shall be compounded monthly and shall be immediately payable "
        "on demand. The Lender's right to charge default interest is without prejudice "
        "to any other right or remedy available to the Lender at law or in equity."
    )

    r.h2("3.3 Financial Covenants")
    r.body(
        "The Borrower undertakes to the Lender that, from the first Interest Payment "
        "Date and at all times thereafter until all amounts outstanding under this "
        "Agreement have been repaid in full, each of the following financial covenants "
        "shall be complied with."
    )
    r.body(
        "Compliance with the financial covenants set out in Clauses 3.3.1 to 3.3.3 "
        "shall be tested semi-annually by reference to the Borrower's most recent "
        "audited annual financial statements or, where applicable, unaudited "
        "management accounts, in each case certified by a director of the Borrower."
    )

    r.h3("3.3.1 Debt Service Coverage Ratio (DSCR)")
    r.body(
        "The Debt Service Coverage Ratio (\"DSCR\"), being the ratio of the Borrower's "
        "Net Operating Income to its Total Debt Service obligations for the relevant "
        "12-month Testing Period, shall not at any time fall below 1.25 times (1.25x)."
    )
    r.body(
        "For this purpose, \"Net Operating Income\" means the Borrower's gross revenue "
        "less operating expenses (excluding interest, tax, depreciation, and "
        "amortisation), and \"Total Debt Service\" means all scheduled principal and "
        "interest payments falling due in respect of the Facility during the Testing Period."
    )

    r.h3("3.3.2 Loan to Value Ratio (LTV)")
    r.body(
        "The Loan to Value Ratio (\"LTV\"), being the ratio of the outstanding principal "
        "amount of the Facility to the Market Value of the secured property, shall not "
        "at any time exceed 65% (sixty-five per cent)."
    )
    r.body(
        "\"Market Value\" shall be determined by reference to the most recent independent "
        "valuation report obtained by the Lender from a qualified valuer approved by "
        "the Lender, which shall be procured at the Borrower's cost no less frequently "
        "than once every 12 months."
    )

    r.h3("3.3.3 Interest Coverage Ratio (ICR)")
    r.body(
        "The Interest Coverage Ratio (\"ICR\"), being the ratio of EBITDA to Net Finance "
        "Charges for the relevant 12-month Testing Period, shall not at any time fall "
        "below 2.00 times (2.00x)."
    )
    r.body(
        "\"EBITDA\" means earnings before interest, taxes, depreciation, and amortisation "
        "as shown in the Borrower's financial statements. \"Net Finance Charges\" means "
        "gross interest expense less interest income for the Testing Period."
    )

    r.h2("3.4 Fees")
    r.body(
        "The Borrower shall pay to the Lender a non-refundable arrangement fee equal "
        "to 1.00% of the total Facility amount on or before the date of first drawdown. "
        "The arrangement fee shall be earned on the date of execution of this Agreement."
    )
    r.body(
        "A commitment fee of 0.50% per annum shall accrue on the daily undrawn and "
        "uncancelled balance of the Facility during the Availability Period, payable "
        "quarterly in arrear on each Interest Payment Date and on the last day of "
        "the Availability Period."
    )

    # ── Section 4 ─────────────────────────────────────────────────────────
    r.h1("4. REPAYMENT")
    r.body(
        "The Borrower shall repay the Facility in accordance with this Clause 4. "
        "All repayments and prepayments shall be made to the Lender's account in "
        "Singapore in immediately available, freely transferable funds."
    )

    r.h2("4.1 Repayment Schedule")
    r.body(
        "The outstanding principal balance of the Facility shall be repaid in 16 "
        "equal quarterly instalments of SGD 3,125,000 each, with the first instalment "
        "due on the date falling 12 months after the date of first drawdown."
    )
    r.body(
        "The Facility shall be fully repaid on the Final Maturity Date, being the date "
        "falling 60 months (5 years) from the date of first drawdown. Any amount "
        "remaining outstanding on the Final Maturity Date shall become immediately "
        "due and payable without further notice or demand."
    )

    r.h2("4.2 Voluntary Prepayment")
    r.body(
        "The Borrower may prepay the whole or any part of the outstanding principal "
        "of the Facility (in a minimum amount of SGD 1,000,000 and in integral "
        "multiples of SGD 500,000) on any Business Day, provided that not less than "
        "5 Business Days' prior written notice is given to the Lender."
    )
    r.body(
        "Any voluntary prepayment shall be accompanied by all accrued and unpaid "
        "interest on the amount prepaid together with a prepayment fee of 1.00% of "
        "the amount prepaid if the prepayment occurs within the first 24 months from "
        "first drawdown, and 0.50% of the amount prepaid thereafter."
    )

    # ── Section 5 ─────────────────────────────────────────────────────────
    r.h1("5. REPRESENTATIONS AND WARRANTIES")
    r.body(
        "The Borrower makes each of the representations and warranties set out in "
        "this Clause 5 to the Lender on the date of this Agreement and on each "
        "Drawdown Date, by reference to the facts and circumstances then existing."
    )

    r.h2("5.1 Status")
    r.body(
        "The Borrower is a private limited company duly incorporated and validly "
        "existing under the laws of Singapore. It has the legal capacity and full "
        "power to own its assets, carry on its business as currently conducted, "
        "and enter into and perform its obligations under this Agreement."
    )

    r.h2("5.2 Financial Condition")
    r.body(
        "The audited financial statements of the Borrower for the financial year ended "
        "31 December 2025, copies of which have been provided to the Lender, were "
        "prepared in accordance with Singapore Financial Reporting Standards (SFRS) "
        "applied consistently and give a true and fair view of the Borrower's financial "
        "position as at that date."
    )
    r.body(
        "No Material Adverse Effect has occurred since the date of the most recent "
        "audited financial statements provided to the Lender. No event, circumstance, "
        "or change has occurred which, with the passage of time or giving of notice, "
        "would reasonably be expected to result in a Material Adverse Effect."
    )

    # ── Section 6 ─────────────────────────────────────────────────────────
    r.h1("6. EVENTS OF DEFAULT")
    r.body(
        "Each of the following constitutes an Event of Default. Upon the occurrence "
        "and continuance of an Event of Default, the Lender may, by written notice "
        "to the Borrower, declare all amounts outstanding under this Agreement to be "
        "immediately due and payable, and enforce any security held by it."
    )

    r.h2("6.1 Non-Payment")
    r.body(
        "The Borrower fails to pay any principal, interest, fee, or other amount "
        "payable by it under this Agreement on its due date in the currency and in "
        "the manner specified herein, unless such failure is attributable solely to "
        "an administrative or technical error and is remedied within 3 Business Days "
        "of the due date."
    )

    r.h2("6.2 Breach of Covenant")
    r.body(
        "The Borrower fails to comply with any financial covenant set out in Clause "
        "3.3, or fails to comply with any other obligation, undertaking, or covenant "
        "under this Agreement (other than a payment obligation), and such failure, "
        "if capable of remedy, is not remedied within 30 days after the earlier of "
        "the Lender giving notice thereof or the Borrower becoming aware of the breach."
    )

    r.h2("6.3 Insolvency")
    r.body(
        "The Borrower is unable or admits inability to pay its debts as they fall due, "
        "or suspends or threatens to suspend making payments in respect of any of its "
        "debts, or the value of its assets is less than the aggregate of its liabilities "
        "(including contingent and prospective liabilities)."
    )
    r.body(
        "A court order is made, or an effective resolution is passed, for the winding "
        "up, dissolution, judicial management, or administration of the Borrower, or "
        "a receiver, manager, liquidator, judicial manager, or similar officer is "
        "appointed over all or any material part of the Borrower's assets or undertaking."
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def generate(output_path: str) -> None:
    r = Renderer()
    build(r)
    r.save(output_path)
    print(f"Generated: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock credit facility agreement PDF.")
    parser.add_argument(
        "--output", "-o",
        default="files/mock_document.pdf",
        help="Output PDF path (default: files/mock_document.pdf)",
    )
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generate(args.output)


if __name__ == "__main__":
    main()
