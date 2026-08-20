# Own the long-context chunking logic instead of pinning gliner2 from git

The released gliner2 (1.3.2) has no long-context APIs; they exist only on git main, unreleased and with a `NameError` bug in `extract_entities_long`, and its merge path requires `format_results=True` — the upstream formatter that this repo deliberately bypasses because it dedupes by lowercased text and drops repeated mentions at distinct positions (see `formatting.py`). We stay on the locked PyPI release and implement chunk splitting, span remapping, and overlap merging ourselves in `services/chunking.py`, keeping our no-dedup mention semantics.

## Considered Options

- **Pin gliner2 from git main** — rejected: unreleased, buggy, forces the upstream dedup formatter.
- **Vendor upstream `chunking.py`** — rejected: it operates on formatted results, so it would still need adaptation; drifting vendored code is worse than ~100 owned lines.
- **Own implementation on the released dependency** — chosen.
