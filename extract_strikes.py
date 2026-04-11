import fitz

doc = fitz.open("your.pdf")
page = doc[0]

print("=== Annotations ===")
annots = list(page.annots())
print(f"Count: {len(annots)}")
for a in annots:
    print(a.type, a.rect, a.colors)

print("\n=== Drawings (first 20) ===")
drawings = page.get_drawings()
print(f"Total drawings: {len(drawings)}")
for d in drawings[:20]:
    rect = d["rect"]
    print(f"  rect={rect} | h={rect.height:.2f} | color={d.get('color')} | fill={d.get('fill')} | type={d.get('type')}")


import fitz

def is_red(color):
    """Check if a color tuple is red or near-red."""
    if not color or len(color) != 3:
        return False
    r, g, b = color
    return r > 0.5 and g < 0.4 and b < 0.4

def get_strikethrough_rects(page, max_line_height=3.0):
    """
    Extract rects that look like strikethrough lines:
    - Very thin horizontal rectangles
    - Red or near-red color
    """
    struck_rects = []
    for d in page.get_drawings():
        rect = d["rect"]
        color = d.get("color") or d.get("fill")

        is_thin = rect.height <= max_line_height
        is_horizontal = rect.width > rect.height * 3  # much wider than tall
        is_red_color = is_red(color)

        if is_thin and is_horizontal and is_red_color:
            struck_rects.append(rect)

    return struck_rects


def extract_text_no_strikeout(pdf_path):
    doc = fitz.open(pdf_path)
    result = []

    for page in doc:
        struck_rects = get_strikethrough_rects(page)
        print(f"Page {page.number + 1}: found {len(struck_rects)} strikethrough(s)")

        words = page.get_text("words")  # (x0, y0, x1, y1, word, ...)
        clean_words = []

        for w in words:
            word_rect = fitz.Rect(w[:4])
            # Expand struck rect vertically to reliably catch the word it crosses
            is_struck = any(
                word_rect.intersects(
                    fitz.Rect(sr.x0, sr.y0 - 8, sr.x1, sr.y1 + 8)
                )
                for sr in struck_rects
            )
            if not is_struck:
                clean_words.append(w[4])

        result.append(" ".join(clean_words))

    return "\n\n".join(result)


print(extract_text_no_strikeout("your.pdf"))


def get_strikethrough_rects_any_color(page):
    struck_rects = []
    for d in page.get_drawings():
        rect = d["rect"]
        is_thin = rect.height <= 3.0
        is_horizontal = rect.width > rect.height * 3

        if is_thin and is_horizontal:
            # Check if the line sits at mid-height of a nearby word
            # by seeing if any word's vertical midpoint aligns with the line
            words = page.get_text("words")
            for w in words:
                word_mid_y = (w[1] + w[3]) / 2  # vertical center of word
                line_mid_y = (rect.y0 + rect.y1) / 2
                if abs(word_mid_y - line_mid_y) < 4:  # line crosses word's middle
                    struck_rects.append(rect)
                    break

    return struck_rects

def is_red(color):
    if not color or len(color) != 3:
        return False
    r, g, b = color
    return r > 0.5 and g < 0.4 and b < 0.4

def is_blue(color):
    if not color or len(color) != 3:
        return False
    r, g, b = color
    return b > 0.5 and r < 0.4 and g < 0.6

def get_strikethrough_rects(page, max_line_height=3.0):
    struck_rects = []
    for d in page.get_drawings():
        rect = d["rect"]
        color = d.get("color") or d.get("fill")

        is_thin = rect.height <= max_line_height
        is_horizontal = rect.width > rect.height * 3
        is_red_color = is_red(color)
        is_blue_color = is_blue(color)

        # Only flag red lines; explicitly skip blue (underlines to preserve)
        if is_thin and is_horizontal and is_red_color and not is_blue_color:
            struck_rects.append(rect)

    return struck_rects

def get_strikethrough_rects(page, max_line_height=3.0):
    struck_rects = []
    words = page.get_text("words")

    for d in page.get_drawings():
        rect = d["rect"]
        color = d.get("color") or d.get("fill")

        is_thin = rect.height <= max_line_height
        is_horizontal = rect.width > rect.height * 3
        is_red_color = is_red(color)

        if not (is_thin and is_horizontal and is_red_color):
            continue

        # Extra guard: confirm the line crosses the middle of a word (not below it)
        line_mid_y = (rect.y0 + rect.y1) / 2
        crosses_midpoint = any(
            w[1] < line_mid_y < w[3]   # line_mid_y falls inside word's y-range
            for w in words
            if fitz.Rect(w[:4]).intersects(fitz.Rect(rect.x0, rect.y0 - 8, rect.x1, rect.y1 + 8))
        )

        if crosses_midpoint:
            struck_rects.append(rect)

    return struck_rects


import fitz

def is_red(color):
    if not color or len(color) != 3:
        return False
    r, g, b = color
    return r > 0.5 and g < 0.4 and b < 0.4

def get_strikethrough_rects(page, max_line_height=3.0):
    struck_rects = []
    words = page.get_text("words")

    for d in page.get_drawings():
        rect = d["rect"]
        color = d.get("color") or d.get("fill")

        is_thin = rect.height <= max_line_height
        is_horizontal = rect.width > rect.height * 3
        is_red_color = is_red(color)

        if not (is_thin and is_horizontal and is_red_color):
            continue

        # Confirm line crosses vertical midpoint of a nearby word (not an underline)
        line_mid_y = (rect.y0 + rect.y1) / 2
        crosses_midpoint = any(
            w[1] < line_mid_y < w[3]
            for w in words
            if fitz.Rect(w[:4]).intersects(rect)  # no padding — exact horizontal overlap only
        )

        if crosses_midpoint:
            struck_rects.append(rect)

    return struck_rects


def extract_text_no_strikeout(pdf_path):
    doc = fitz.open(pdf_path)
    result = []

    for page in doc:
        struck_rects = get_strikethrough_rects(page)

        # Use rawdict for char-level bounding boxes
        blocks = page.get_text("rawdict")["blocks"]
        clean_chars = []
        current_line_chars = []
        last_line_y = None

        for block in blocks:
            if block["type"] != 0:  # skip non-text blocks
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    for char in span["chars"]:
                        char_rect = fitz.Rect(char["bbox"])
                        char_mid_y = (char_rect.y0 + char_rect.y1) / 2

                        is_struck = any(
                            # Horizontal overlap: char must be under the struck rect x-range
                            char_rect.x0 < sr.x1 and char_rect.x1 > sr.x0
                            # Vertical: struck line must cross the char's vertical midpoint
                            and sr.y0 - 2 < char_mid_y < sr.y1 + 2
                            for sr in struck_rects
                        )

                        if not is_struck:
                            # Track line breaks by y-position
                            line_y = round(char_rect.y0, 1)
                            if last_line_y is not None and abs(line_y - last_line_y) > 2:
                                clean_chars.append("\n")
                            last_line_y = line_y
                            clean_chars.append(char["c"])

        result.append("".join(clean_chars))

    return "\n\n".join(result)


print(extract_text_no_strikeout("your.pdf"))