# Run Ray Serve without a `uv run` ancestor

Ray's uv runtime-env hook detects when the driver runs under `uv run` and starts every worker via `uv run --python <ver> python` inside an extracted copy of the repo's working_dir. Each worker's `uv run` then builds a full project venv (torch et al.) inside `/tmp/ray/...`, which blew the `/tmp` quota (`os error 122`) and crashed worker startup entirely. `mise dev` therefore syncs first and launches the venv binary directly (`uv sync --quiet && .venv/bin/ray serve run ...`): workers then run the venv python against the packaged source (`.rayignore` keeps `.venv` out) with no per-worker venv creation.

## Consequences

- Don't wrap the serve command in `uv run` (also in Docker/CI); the venv must already exist.
- GLiNER2 is not picklable (HF tokenizers lru-cache), so Ray tasks cannot receive the model via the object store — task workers build and cache their own `GlinerService` (one model copy per worker, loaded once).
