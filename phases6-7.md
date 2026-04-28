## Step 6 — Search for each term separately (in depth)

This is handled by `retrieve_for_term`. It runs for **every single term** the LLM extracted. So if 8 terms came out, this whole step runs 8 times.

For each term, two searches happen in parallel:

---

### Search A: Meaning-based (semantic) search

```
for doc, score in vector_store.similarity_search_with_score(term, k=top_k):
```

The term (e.g. `"Final Maturity Date"`) is converted into an embedding — the same kind of numbers-that-represent-meaning used for the chunks. The vector store then finds the *k* chunks (default 5) whose embeddings are closest in meaning to the term.

**What "closest in meaning" actually means:**

The similarity score is a number between 0 and 1. A score of 0.9 means the chunk is talking about almost exactly the same thing as the term. A score of 0.35 means there's a loose connection. The vector store returns the top 5 by this score.

**The problem with relying on this alone:**

Imagine the definitions section has an entry literally headed *"Final Maturity Date"* followed by its full legal definition. That chunk should obviously be retrieved. But its body text might be dominated by cross-references, conditions, and boilerplate — so when it gets embedded, the resulting number-vector is pulled toward all that surrounding language, not just the term name. It might end up ranked 7th or 8th in similarity and get cut off at top-5.

This is why Search B exists.

---

### Search B: Heading keyword match (injection)

```
term_lower = term.lower()for chunk in chunks:    if term_lower in chunk.heading.lower() and cid not in seen:        seen[cid] = (_chunk_to_doc(chunk), 1.0)
```

This loops through every single chunk and checks: does this chunk's **heading** contain the term as a substring?

If yes — the chunk is added with a score of **1.0** (the maximum possible), regardless of what the semantic search thought of it.

This is a hard guarantee: if a section is literally titled something containing your search term, it will always be included. It bypasses the embedding similarity entirely for heading matches.

**Why score 1.0 specifically?**

Because in a legal document, if the heading says *"Final Maturity Date"* — that section *is* the definition of Final Maturity Date. There is no ambiguity. Giving it 1.0 ensures it will survive the score threshold filter later and rank near the top.

---

### Deduplication within the term's results

```
if cid not in seen or score > seen[cid][1]:    seen[cid] = (doc, score)
```

Both searches might return the same chunk. When that happens, only the **higher score** is kept. So if semantic search found a chunk with score 0.72, and heading injection also matched it at 1.0, it's stored as 1.0.

At the end of `retrieve_for_term`, you have a clean list of `(chunk, score)` pairs for that one term — no duplicates, each chunk at its best possible score.

---

### Concrete example

Say the terms are `["Final Maturity Date", "Extension Option"]` and `top_k = 3`.

**For "Final Maturity Date":**

| Source            | Chunk                                    | Score |
| ----------------- | ---------------------------------------- | ----- |
| Semantic          | Clause 19 (Extension Option)             | 0.81  |
| Semantic          | Clause 3.1 (Purpose)                     | 0.62  |
| Semantic          | Schedule 2 (Rate Terms)                  | 0.58  |
| Heading injection | 1.1 Definitions → "Final Maturity Date" | 1.00  |

Result after dedup: 4 chunks

**For "Extension Option":**

| Source            | Chunk                                    | Score                 |
| ----------------- | ---------------------------------------- | --------------------- |
| Semantic          | Clause 19 (Extension Option)             | 0.93                  |
| Semantic          | 1.1 Definitions → "Final Maturity Date" | 0.71                  |
| Semantic          | Clause 22 (Conditions Precedent)         | 0.54                  |
| Heading injection | Clause 19 (Extension Option)             | 1.00 → replaces 0.93 |

Result after dedup: 3 chunks (Clause 19 kept at 1.0)

---

## Step 7 — Merge all results together (in depth)

Now you have separate result lists from all 8 (or however many) terms. The merge step collapses these into one pool.

```
merged: dict[int, tuple[Document, float, list[str]]] = {}
for term in terms:    for doc, score in retrieve_for_term(term, ...):        if score < score_threshold:            continue                          # drop low-confidence results here        cid = doc.metadata["chunk_id"]        if cid not in merged:            merged[cid] = (doc, score, [term])        else:            old_doc, old_score, old_terms = merged[cid]            updated_terms = old_terms if term in old_terms else old_terms + [term]            merged[cid] = (old_doc, max(old_score, score), updated_terms)
```

The merge dictionary is keyed by `chunk_id`. As each term's results are processed:

---

### First encounter of a chunk

```
merged[cid] = (doc, score, [term])
```

The chunk is stored with:

* Its text content
* The score this term gave it
* A list containing just this one term

---

### Chunk seen again (from a different term's search)

```
updated_terms = old_terms if term in old_terms else old_terms + [term]merged[cid] = (old_doc, max(old_score, score), updated_terms)
```

Two things are updated:

1. **Score** — `max(old_score, score)` keeps whichever score is higher. If "Final Maturity Date" found this chunk at 0.81, and "Extension Option" found the same chunk at 0.93, the stored score becomes 0.93. The chunk is only ever represented at its best score.
2. **Matched terms list** — the new term is appended. This list is what drives the boost in the next step — so tracking it accurately is critical.

---

### What the merged dictionary looks like after all terms

Continuing the example:

| Chunk                                  | Best score | Matched terms                                   |
| -------------------------------------- | ---------- | ----------------------------------------------- |
| Clause 19 (Extension Option)           | 1.00       | `["Final Maturity Date", "Extension Option"]` |
| 1.1 Definitions → Final Maturity Date | 1.00       | `["Final Maturity Date", "Extension Option"]` |
| Clause 3.1 (Purpose)                   | 0.62       | `["Final Maturity Date"]`                     |
| Schedule 2 (Rate Terms)                | 0.58       | `["Final Maturity Date"]`                     |
| Clause 22 (Conditions Precedent)       | 0.54       | `["Extension Option"]`                        |

Notice:

* Clause 19 and the definitions entry both matched **two terms** — they will get the multi-term boost in Step 8
* Clause 3.1, Schedule 2, and Clause 22 only matched **one term** — no boost

---

### The threshold filter (also happens here)

```
if score < score_threshold:    continue
```

This check runs **before** a chunk enters the merged dictionary. If a term retrieved a chunk at score 0.28 (below the default threshold of 0.35), it's silently skipped — it never even gets a chance to accumulate matched terms or score. This keeps genuinely irrelevant chunks out of the pool entirely rather than letting them creep in just because they matched a term at low confidence.

---

### Summary of what Steps 6 and 7 produce together

You started with one free-text statement. You now have a **single flat pool** of unique chunks, each one carrying:

* The highest similarity score any term gave it
* The full list of terms that considered it relevant

This is the input to the boost and ranking in Step 8.
