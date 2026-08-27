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
import logging
import os
from collections.abc import AsyncIterable

import torch
from fastapi import FastAPI, Header, HTTPException
from ray import serve

from dcc_gliner_api.models import (
    BatchExtractEntitiesRequest,
    ExtractEntitiesRequest,
    TaskAccepted,
    TaskState,
)
from dcc_gliner_api.models.entities import ExtractEntitiesBatchResponse, ExtractEntitiesResponse
from dcc_gliner_api.services.chunking import split_text_into_chunks
from dcc_gliner_api.services.gliner_service import GlinerService
from dcc_gliner_api.services.task_store import Task, TaskStore

# Ray Serve configures this logger, so messages reach the replica log with the
# request id attached; a module logger would be filtered at WARNING.
logger = logging.getLogger("ray.serve")

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


def _num_gpus() -> float:
    """GPUs to reserve per replica.

    Ray hides GPUs from an actor that did not ask for one, so without this the
    replica sees no CUDA device however the model is loaded.
    """
    configured = os.environ.get("GLINER_NUM_GPUS")
    if configured is not None:
        return float(configured)

    return 1.0 if torch.cuda.is_available() else 0.0


@serve.deployment(ray_actor_options={"num_gpus": _num_gpus()})
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
    async def extract_entities_async(
        self,
        request: ExtractEntitiesRequest,
        x_correlation_id: str | None = Header(default=None),
    ) -> TaskAccepted:
        """Accept the work and return at once, so a long scan cannot time out."""
        # Logged on arrival: if a caller reports an error but nothing shows up
        # here, the request was rejected before it ever reached this service.
        logger.info(
            "extract_entities/async received: %d chars, %d chunks, correlation_id=%s",
            len(request.text),
            len(split_text_into_chunks(request.text)),
            x_correlation_id or "-",
        )

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

        task_id = self.tasks.submit(run).id
        logger.info("extract_entities/async accepted: task_id=%s", task_id)
        return TaskAccepted(task_id=task_id)

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


app = GLiNER2Deployment.bind()  # added by @serve.deployment
