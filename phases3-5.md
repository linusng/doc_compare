## Step 3 — Cut the document into sections (in depth)

This is two sub-steps that run back to back: first  **detect which blocks are headings** , then  **group everything under them** .

---

### Sub-step 3a: Detect headings

```
font_sizes = [b.font_size for b in blocks if b.font_size > 0]median_size = statistics.median(font_sizes)
is_heading = (    block.font_size > median_size * 1.1    or (block.is_bold and len(block.text) < 120))
```

The script never looks for keywords like "Section" or "Clause". Instead it uses two purely visual signals:

**Signal 1 — Font size.** It computes the median font size across every block in the whole document. Any block whose font size is more than 10% above that median is flagged as a heading. In a typical legal document, body text is 10–11pt and headings are 12–14pt, so the threshold catches them cleanly without any hardcoding.

**Signal 2 — Bold + short text.** Even if a heading uses the same font size as body text, it's usually bold and short (clause labels like `"1.1 Definitions"` rarely exceed 120 characters). A block that satisfies both conditions is also flagged.

This is fully language-agnostic — it works on Japanese, Arabic, German, anything — because it's reacting to visual formatting, not words.

---

### Sub-step 3b: Group into SectionChunks

```
for block in blocks:    if block.is_heading:        saved = flush(len(chunks))   # close off previous section        if saved:            chunks.append(saved)        current_heading = block.text        current_body = []        current_pages = {block.page}    else:        current_body.append(block.text)        current_pages.add(block.page)
```

The blocks (now in document order) are processed one by one:

* When a **heading block** is encountered, everything accumulated so far is saved as a completed `SectionChunk`, and a new one is started with that heading.
* When a **body block** is encountered, it's simply appended to the current body and its page number is recorded.

Each resulting `SectionChunk` carries:

* `heading` — the heading text that opens the section
* `content` — all body text beneath it
* `pages` — the set of pages the section spans
* `heading_level` — detected from the numbering pattern (`1.` = level 1, `1.1` = level 2, etc.)
* `chunk_id` — a sequential integer used everywhere for deduplication

**Concrete example of what comes out:**

| chunk_id | heading                              | pages        |
| -------- | ------------------------------------ | ------------ |
| 0        | (preamble text before first heading) | [1]          |
| 1        | 1. Definitions and Interpretation    | [3, 4, 5, 6] |
| 2        | 1.1 Definitions                      | [3, 4, 5]    |
| 3        | 1.2 Construction                     | [6]          |
| 4        | 2. The Facility                      | [7]          |
| …       | …                                   | …           |
| 41       | Schedule 2 Compounded Rate Terms     | [88, 89, 90] |

---

### Sub-step 3c: Split chunks that are too large

```
splitter = RecursiveCharacterTextSplitter(    chunk_size=max_tokens,           # 8192    chunk_overlap=chunk_overlap,     # 50 tokens    length_function=token_len,    separators=["\n\n", "\n", ". ", " ", ""],)
```

The embedding model (BGE-M3) has a hard limit of  **8,192 tokens** . A definitions section spanning 10 pages will blow past that limit — feeding it in as-is would silently truncate it or cause an error.

For every chunk that's too big, the splitter breaks it into smaller pieces. The key details:

* **`length_function=token_len`** — it measures size using the actual BGE-M3 tokeniser, not characters or words, so the split boundary is exact for this model
* **`separators=["\n\n", "\n", ". ", " ", ""]`** — it tries to split at the most natural boundary first (paragraph break), falling back to sentence, then word, then character only if necessary
* **50-token overlap** — each sub-chunk shares 50 tokens with the previous one, so a sentence that straddles a boundary isn't cut off mid-thought in both pieces

A chunk that gets split is renamed:

> `"1.1 Definitions"` → `"1.1 Definitions (part 1)"`, `"1.1 Definitions (part 2)"`, …

The `(part N)` suffix is stripped back out later whenever two chunks need to be compared by heading — the `_base_heading()` function handles this throughout the codebase.

After splitting, all `chunk_id` values are reassigned sequentially so every chunk has a unique ID.

---

## Step 4 — Build the vector store (in depth)

```
docs = [    Document(        page_content=chunk.full_text,   # heading + body together        metadata={            "chunk_id": chunk.chunk_id,            "heading": chunk.heading,            "pages": chunk.pages,            ...        },    )    for chunk in chunks]
vector_store = InMemoryVectorStore(embedding=embeddings)vector_store.add_documents(docs)
```

---

### What gets embedded

Notice `chunk.full_text`, not just `chunk.content`:

```
@propertydef full_text(self) -> str:    return f"{self.heading}\n{self.content}".strip()
```

The heading is prepended to the body text before embedding. This matters because a chunk headed `"19. Extension Option"` followed by dense legal boilerplate will embed differently (and more relevantly) with its heading included than without it. The heading tells the embedding model what the body text is  *about* .

---

### What the embedding model does

BGE-M3 converts the full text of each chunk into a list of ~1024 numbers. These numbers are not random — they're positioned in a mathematical space where chunks with similar *meaning* end up close together, regardless of whether they use the same words.

For example:

* A chunk about `"Final Maturity Date"` and a chunk about `"maturity of the loan"` end up nearby
* A chunk about `"governing law"` ends up far from both

This is what makes searching by meaning possible in Step 6.

---

### The vector store

`InMemoryVectorStore` holds all these number-vectors in RAM (no database, no disk). When you search it, it computes the cosine similarity between your query's embedding and every stored chunk's embedding and returns the top matches.

**Cosine similarity** is essentially the angle between two vectors — if two vectors point in the same direction, the angle is 0 and the score is 1.0 (identical meaning). If they point in completely different directions, the score is 0.

The vector store retains all the metadata alongside each vector — chunk_id, heading, pages — so that when a chunk is returned by search, you know exactly which section it came from and what pages it covers.

---

## Step 5 — Extract key terms from the statement (in depth)

```
prompt = ChatPromptTemplate.from_messages([    ("system", (        "Return ONLY a JSON array of strings, 5–15 terms, most specific first.\n"        'Example: ["Final Maturity Date", "Extension Option", ...]'    )),    ("human", "{statement}"),])
result = (prompt | llm).invoke({"statement": statement})
```

---

### Why decompose the statement at all

If you embed the full statement and search with it once, you get one ranked list of chunks — and the top result will be whichever section is most similar to the *overall* statement. That's usually the main operative clause (e.g. the Extension Option clause), which is right, but it tends to crowd out supporting definitions and cross-references that are equally necessary to understand the full picture.

By breaking the statement into individual terms and searching for each one separately, you fire a focused spotlight at each concept independently. The definitions entry for `"Final Maturity Date"` might score poorly against the full statement but scores highly when searched alone.

---

### What the LLM is instructed to extract

The prompt gives the LLM explicit categories to look for:

| Category                  | Example                                                   |
| ------------------------- | --------------------------------------------------------- |
| Capitalised defined terms | `"Final Maturity Date"`, `"Borrower"`, `"Facility"` |
| Named mechanisms          | `"Extension Option"`, `"Utilisation Request"`         |
| Section-level concepts    | `"Repayment"`, `"Interest"`                           |
| Financial terms           | `"Margin"`, `"Commitment Period"`                     |

The instruction `"most specific first"` ensures that precise multi-word terms like `"Final Maturity Date"` appear before broad single words like `"Borrower"`. This matters because specific terms produce tighter, more accurate retrieval results — they're processed first and already cover the important chunks before the broader terms run.

The LLM is also told to **exclude** generic words like "the", "option", or "date" standing alone — these would produce noise in the semantic search.

---

### Parsing the output

```
m = re.search(r'\[.*?\]', raw, re.DOTALL)if m:    terms = json.loads(m.group())
```

The LLM is asked to return a JSON array. The regex `\[.*?\]` extracts just the array even if the model wraps it in explanation text (which some models do despite being told not to). If JSON parsing fails entirely, there's a fallback that splits on commas and newlines and strips quote characters — so the step degrades gracefully rather than crashing.

---

### What comes out

For the statement:

> *"Extension Option: the Final Maturity Date can be extended up to two times, one year each from the Original Maturity Date."*

The output might be:

```
[  "Final Maturity Date",  "Extension Option",  "Original Maturity Date",  "Borrower",  "Facility tenor",  "utilisation date",  "Maturity Date",]
```

Each of these then becomes an independent search query in Step 6.
