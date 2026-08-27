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

import asyncio
import json
import os
from collections.abc import AsyncIterable

from fastapi import FastAPI, HTTPException
from ray import serve

from dcc_gliner_api.models import (
    BatchExtractEntitiesRequest,
    ExtractEntitiesRequest,
    TaskAccepted,
    TaskState,
)
from dcc_gliner_api.models.entities import ExtractEntitiesBatchResponse, ExtractEntitiesResponse
from dcc_gliner_api.services.gliner_service import GlinerService
from dcc_gliner_api.services.task_store import Task, TaskStore

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
        self.tasks = TaskStore()

    @app.post("/extract_entities", summary="Entity extraction (full-document chunk scan)")
    def extract_entities(self, request: ExtractEntitiesRequest):
        return self.service.extract_entities(
            request.text,
            request.entity_types,
            threshold=request.threshold,
        )

    @app.post(
        "/extract_entities/async",
        summary="Submit an entity extraction and poll for it",
        status_code=202,
    )
    async def extract_entities_async(self, request: ExtractEntitiesRequest) -> TaskAccepted:
        """Accept the work and return at once, so a long scan cannot time out."""

        async def run(task: Task) -> ExtractEntitiesResponse:
            def report(done: int, total: int) -> None:
                task.progress = done / total if total else None
                task.touch()

            # The model call is blocking, so it runs off the event loop and the
            # status endpoint stays responsive while it works.
            return await asyncio.to_thread(
                self.service.extract_entities,
                request.text,
                request.entity_types,
                threshold=request.threshold,
                on_progress=report,
            )

        return TaskAccepted(task_id=self.tasks.submit(run).id)

    @app.get("/task/{task_id}", summary="Status of a submitted task")
    async def task_state(self, task_id: str) -> TaskState:
        task = self.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Unknown task")

        return TaskState(
            task_id=task.id,
            status=task.status,
            progress=task.progress,
            resource_id=task.resource_id,
            error=task.error,
        )

    @app.get("/resource/{resource_id}", summary="Collect a finished result (once)")
    async def resource(self, resource_id: str) -> ExtractEntitiesResponse:
        """Hand the result over and drop it, so results do not pile up."""
        found, result = self.tasks.take_resource(resource_id)
        if not found:
            raise HTTPException(status_code=404, detail="Unknown or already collected resource")
        return result

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
