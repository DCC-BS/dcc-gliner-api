"""GLiNER2 API exposing all model capabilities over HTTP via Ray Serve.

Capabilities (https://github.com/fastino-ai/GLiNER2):
- Entity extraction (with descriptions, confidence, spans) as a full-document
  chunk scan with overlapping chunks
- Relation extraction
- Text classification (single/multi-label)
- Structured JSON extraction (field specs, choices)
- Multi-task schema extraction (compose all of the above, incl. regex validators)
- Batch variants of all of the above

This layer only translates HTTP requests into service calls and shapes
responses; all model interaction lives in ``services.gliner_service``.
"""

import json
import os
from collections.abc import AsyncIterable

from fastapi import FastAPI
from ray import serve

from dcc_gliner_api.models import (
    BatchExtractEntitiesRequest,
    ExtractEntitiesRequest,
)
from dcc_gliner_api.models.entities import ExtractEntitiesBatchResponse
from dcc_gliner_api.services.gliner_service import GlinerService

app = FastAPI(
    title="GLiNER2 API",
    description="Unified schema-based information extraction and text classification "
    "using GLiNER2: entities, relations, classification, structured data, "
    "multi-task schemas, and batch processing.",
    version="0.5.0",
)


def _ndjson(results):
    for result in results:
        yield json.dumps(result, ensure_ascii=False) + "\n"


@serve.deployment
@serve.ingress(app)
class GLiNER2Deployment:
    def __init__(self):
        if os.getenv("RAY_DEBUG", "0") == "1":
            import debugpy  # noqa: T100

            debugpy.listen(("127.0.0.1", 5678))  # noqa: T100
            print("Waiting for debugger on 127.0.0.1:5678...")
            debugpy.wait_for_client()  # noqa: T100

        self.service = GlinerService()

    @app.post("/extract_entities", summary="Entity extraction (full-document chunk scan)")
    def extract_entities(self, request: ExtractEntitiesRequest):
        entities = self.service.extract_entities(
            request.text,
            request.entity_types,
            threshold=request.threshold,
        )

        d = {}
        d["entities"] = entities

        return d

    @app.post(
        "/batch_extract_entities",
        summary="Batch entity extraction (chunk scan per text, streamed as NDJSON)",
    )
    async def batch_extract_entities(
        self, request: BatchExtractEntitiesRequest
    ) -> AsyncIterable[ExtractEntitiesBatchResponse]:
        """Stream one JSON line per document as each batch window completes."""
        for doc_result in self.service.batch_extract_entities(
            request.texts,
            request.entity_types,
            batch_size=request.batch_size,
            threshold=request.threshold,
        ):
            yield doc_result


app = GLiNER2Deployment.bind()  # ty: ignore[unresolved-attribute]  # added by @serve.deployment
