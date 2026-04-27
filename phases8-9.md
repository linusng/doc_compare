* Merge All Term Results
* Multi-Term Boost
* Filter by Score Threshold
* Group by section sort by page

These four steps all happen across two functions: `gather_all_relevant` and `build_collated_context`. Let me walk through each one concretely.

---

## Setup: what comes in

Before these steps, per-term retrieval has already run. For each term (e.g. `"Final Maturity Date"`, `"Extension Option"`, `"Borrower"`), the pipeline ran a separate semantic search and returned a list of `(document_chunk, similarity_score)` pairs.

The problem:  **multiple terms can return the same chunk** , and some chunks are more important than others because they cover several terms at once.

---

## Step 1 — Merge All Term Results

```
merged: dict[int, tuple[Document, float, list[str]]] = {}
for term in terms:    for doc, score in retrieve_for_term(term, ...):        cid = doc.metadata["chunk_id"]        if cid not in merged:            merged[cid] = (doc, score, [term])       # first time seeing this chunk        else:            old_doc, old_score, old_terms = merged[cid]            updated_terms = old_terms if term in old_terms else old_terms + [term]            merged[cid] = (old_doc, max(old_score, score), updated_terms)
```

Each chunk has a unique `chunk_id`. As the loop runs over every term:

* **First encounter** of a chunk → store it with its score and which term found it
* **Seen before** → don't duplicate it; instead update two things:
  * Keep the **highest score** seen so far (`max(old_score, score)`)
  * Append the new term to the chunk's **matched_terms list**

After this loop, `merged` is a flat dictionary: one entry per unique chunk, carrying the best raw score and the full list of terms that matched it.

**Example state after merging:**

| Chunk                           | Best raw score | Matched terms                                         |
| ------------------------------- | -------------- | ----------------------------------------------------- |
| Clause 19 (Extension Option)    | 0.87           | `["Extension Option", "Final Maturity Date"]`       |
| Definition: Final Maturity Date | 0.91           | `["Final Maturity Date", "Original Maturity Date"]` |
| Clause 3.1 (Purpose)            | 0.61           | `["Borrower"]`                                      |

---

## Step 2 — Multi-Term Boost

```
boosted = [    (doc, score * (len(matched_terms) ** 0.5), matched_terms)    for doc, score, matched_terms in merged.values()]
```

The raw score only tells you how similar a chunk is to **one** term. But a chunk that covers **multiple** terms from your statement is more useful — it's the part of the document that ties the concepts together.

The boost formula:  **`boosted_score = raw_score × √N`** , where N = number of distinct terms matched.

Using square root (rather than multiplying by N directly) keeps the boost proportional but not explosive:

| N terms matched | Multiplier |
| --------------- | ---------- |
| 1               | ×1.00     |
| 2               | ×1.41     |
| 3               | ×1.73     |
| 4               | ×2.00     |

Applying to the example:

| Chunk                           | Raw score | N | Boosted score                |
| ------------------------------- | --------- | - | ---------------------------- |
| Clause 19                       | 0.87      | 2 | 0.87 × 1.41 =**1.23** |
| Definition: Final Maturity Date | 0.91      | 2 | 0.91 × 1.41 =**1.28** |
| Clause 3.1 (Purpose)            | 0.61      | 1 | 0.61 × 1.00 =**0.61** |

The definitions clause edges ahead because it's both highly similar and multi-term. Clause 3.1 drops to the bottom because it only matched one term.

After this, the list is  **sorted descending by boosted score** .

---

## Step 3 — Filter by Score Threshold

```
if score < score_threshold:    continue
```

This actually happens *inside* `gather_all_relevant`, **before** the chunk enters the `merged` dict — so it filters on the **raw score** at retrieval time, not the boosted score.

Default threshold: `0.35`.

Any chunk whose raw similarity score is below 0.35 for *every* term that could have found it is dropped entirely. This removes chunks that were returned by the vector store simply because they were the "least bad" match, not because they're actually relevant.

Practically: if your statement is about extension options and the vector store returns a chunk about governing law clauses (low similarity), it gets discarded here rather than polluting the collated context.

---

## Step 4 — Group by Section, Sort by Page

```
by_section: dict[str, list[...]] = defaultdict(list)for doc, score, terms in ranked[:max_chunks]:    heading = _base_heading(doc.metadata.get("heading", "(no heading)"))    by_section[heading].append((doc, score, terms))
sorted_sections = sorted(by_section.items(), key=lambda kv: first_page(kv[1]))
```

At this point you have up to `max_chunks` (default 25) chunks, ranked by boosted score. But dumping them in score order into the context would be jarring — a reader (human or LLM) expects document content to flow in  **document order** , not relevance order.

Two groupings happen:

**Group by section heading** (`_base_heading` strips part numbers like "19.1", "19.2" → "19" so sub-clauses of the same section collapse together under one heading block)

**Sort sections by their earliest page number** so the context reads chronologically through the document — definitions (page 3) appear before the extension clause (page 42), which is how the actual agreement reads.

Within each section, chunks are also sorted by their start page for the same reason.

---

## End result

The `combined_context` string you get out looks like a clean, page-ordered, section-grouped digest of everything in the document that's relevant to your statement — ready to hand to the synthesis step or directly to another LLM.
