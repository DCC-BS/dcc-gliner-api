"""Ray remote tasks for multi-task schema extraction.

The GLiNER2 model is not picklable (HF tokenizers lru-cache), so tasks cannot
receive it through the object store. Instead each Ray worker lazily builds and
caches its own ``GlinerService`` on first task and reuses it for every later
task on that worker. Documents in a batch run as concurrent tasks across
cluster CPUs while the deployment actor only collects and streams results.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import ray

from dcc_gliner_api.services.gliner_service import GlinerService


@lru_cache(maxsize=1)
def _service() -> GlinerService:
    return GlinerService()


@ray.remote
def extract_schema(
    text: str,
    spec: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    """Extract one document against one schema spec; returns the raw result."""
    service = _service()
    return service.extract(text, service.build_schema(spec), **options)
