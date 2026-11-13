# Chunks are bounded in subword tokens, and every chunk is measured before it is scanned

A scan on an 11 GiB UAT card ran out of memory on a document we could not inspect. Metering public Grosser Rat documents and generated ones showed why: the scan was sized from the first chunk's sequence length alone, and chunks were bounded only in words (384), while the encoder reads subword tokens and its attention memory grows with their square. A cover page of prose followed by links, base64, hashes or text whose spaces were lost in extraction gave later chunks of 4,000–14,000 tokens against a first chunk of ~740 — a batch planned for eight prose chunks then asked for up to 24 GiB at once. No memory leaked: allocation returned to the baseline after every document, including after each out-of-memory error.

We now cut any chunk that reads to more than `MAX_CHUNK_TOKENS` (1024) subword tokens, splitting a single word over the bound at character boundaries, and size every model call for the longest chunk it carries. A chunk's length is the schema's (measured once per scan) plus the per-word subword counts of its text, which matches the collator exactly or overestimates by a couple of tokens. Prose never reaches the bound (384 words read to roughly 750–950 tokens), so ordinary documents are chunked and batched as before.

## Considered Options

- **Measure every chunk but keep word-only chunks** — rejected: a single chunk of 8,000+ tokens does not fit an 11 GiB card even alone, whatever the batch.
- **Truncate long chunks** — rejected: silently drops text, the failure ADR 0002 removed.
- **Plan for the worst case (schema + bound) everywhere** — rejected: about halves the batch for ordinary documents with a large schema, for no gain on them.
