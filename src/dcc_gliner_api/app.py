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

import ray
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ray import serve

from .formatting import (
    classification_tasks_of,
    relation_names_of,
    shape_classification_result,
    shape_extraction_result,
    shape_relation_result,
)
from .models import (
    BatchClassifyTextRequest,
    BatchExtractEntitiesRequest,
    BatchExtractJsonRequest,
    BatchExtractRelationsRequest,
    BatchExtractRequest,
    ClassifyTextRequest,
    ExtractEntitiesRequest,
    ExtractJsonRequest,
    ExtractRelationsRequest,
    ExtractRequest,
)
from .services.gliner_service import GlinerService
from .services.tasks import extract_schema

app = FastAPI(
    title="GLiNER2 API",
    description="Unified schema-based information extraction and text classification "
    "using GLiNER2: entities, relations, classification, structured data, "
    "multi-task schemas, and batch processing.",
    version="0.5.0",
)

DEFAULT_LABELS = [
    "person",
    "organisation",
    "adresse",
    "geburtsdatum",
    "datum",
    "ahv-nummer",
    "aktenzeichen",
    "telefonnummer",
    "e-mail-adresse",
    "iban",
    "geldbetrag",
]


def _ndjson(results):
    for result in results:
        yield json.dumps(result, ensure_ascii=False) + "\n"


@serve.deployment
@serve.ingress(app)
class GLiNER2Deployment:
    def __init__(self):
        self.service = GlinerService()

    @app.post(
        "/extract_entities", summary="Entity extraction (full-document chunk scan)"
    )
    def extract_entities(self, request: ExtractEntitiesRequest):
        return self.service.extract_entities(
            request.text,
            request.entity_types,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
        )

    @app.post("/extract_relations", summary="Relation extraction")
    def extract_relations(self, request: ExtractRelationsRequest):
        raw = self.service.extract_relations(
            request.text,
            request.relation_types,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )
        return shape_relation_result(raw, relation_names_of(request.relation_types))

    @app.post("/classify_text", summary="Text classification (single/multi-label)")
    def classify_text(self, request: ClassifyTextRequest):
        raw = self.service.classify_text(
            request.text,
            request.tasks,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            max_len=request.max_len,
        )
        return shape_classification_result(
            raw, list(request.tasks), request.include_confidence
        )

    @app.post("/extract_json", summary="Structured data extraction")
    def extract_json(self, request: ExtractJsonRequest):
        return self.service.extract_json(
            request.text,
            request.structures,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/extract", summary="Multi-task schema extraction (Ray task)")
    def extract(self, request: ExtractRequest):
        raw = ray.get(
            extract_schema.remote(
                request.text,
                request.schema,
                {
                    "threshold": request.threshold,
                    "include_confidence": request.include_confidence,
                    "include_spans": request.include_spans,
                    "max_len": request.max_len,
                },
            )
        )
        return shape_extraction_result(
            raw,
            request.include_confidence,
            classification_tasks_of(request.schema),
            relation_names_of(request.schema.get("relations", [])),
        )

    @app.post(
        "/batch_extract_entities",
        summary="Batch entity extraction (chunk scan per text, streamed as NDJSON)",
    )
    def batch_extract_entities(self, request: BatchExtractEntitiesRequest):
        """Stream one JSON line per document as each batch window completes."""
        results = self.service.iter_batch_extract_entities(
            request.texts,
            request.entity_types,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
        )
        return StreamingResponse(_ndjson(results), media_type="application/x-ndjson")

    @app.post("/batch_extract_relations", summary="Batch relation extraction")
    def batch_extract_relations(self, request: BatchExtractRelationsRequest):
        raw = self.service.batch_extract_relations(
            request.texts,
            request.relation_types,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )
        names = relation_names_of(request.relation_types)
        return [shape_relation_result(r, names) for r in raw]

    @app.post("/batch_classify_text", summary="Batch text classification")
    def batch_classify_text(self, request: BatchClassifyTextRequest):
        raw = self.service.batch_classify_text(
            request.texts,
            request.tasks,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            max_len=request.max_len,
        )
        tasks = list(request.tasks)
        return [
            shape_classification_result(r, tasks, request.include_confidence)
            for r in raw
        ]

    @app.post("/batch_extract_json", summary="Batch structured data extraction")
    def batch_extract_json(self, request: BatchExtractJsonRequest):
        return self.service.batch_extract_json(
            request.texts,
            request.structures,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post(
        "/batch_extract",
        summary=(
            "Batch multi-task schema extraction "
            "(one Ray task per document, streamed as NDJSON)"
        ),
    )
    def batch_extract(self, request: BatchExtractRequest):
        """Run one Ray task per document concurrently; stream results in order."""
        if isinstance(request.schemas, dict):
            specs = [request.schemas] * len(request.texts)
        else:
            specs = request.schemas
        options = {
            "threshold": request.threshold,
            "include_confidence": request.include_confidence,
            "include_spans": request.include_spans,
            "max_len": request.max_len,
        }
        refs = [
            extract_schema.remote(text, spec, options)
            for text, spec in zip(request.texts, specs)
        ]

        def results():
            for ref, spec in zip(refs, specs):
                raw = ray.get(ref)
                yield shape_extraction_result(
                    raw,
                    request.include_confidence,
                    classification_tasks_of(spec),
                    relation_names_of(spec.get("relations", [])),
                )

        return StreamingResponse(_ndjson(results()), media_type="application/x-ndjson")


app = GLiNER2Deployment.bind()  # ty: ignore[unresolved-attribute]  # added by @serve.deployment
