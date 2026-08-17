"""GLiNER2 API exposing all model capabilities over HTTP via Ray Serve.

Capabilities (https://github.com/fastino-ai/GLiNER2):
- Entity extraction (with descriptions, confidence, spans)
- Relation extraction
- Text classification (single/multi-label)
- Structured JSON extraction (field specs, choices)
- Multi-task schema extraction (compose all of the above, incl. regex validators)
- Batch variants of all of the above
"""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI

from gliner2 import GLiNER2, RegexValidator

from ray import serve

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
    ValidatorSpec,
)

MODEL_ID = os.environ.get("GLINER_MODEL", "fastino/gliner2-base-v1")

app = FastAPI(
    title="GLiNER2 API",
    description="Unified schema-based information extraction and text classification "
    "using GLiNER2: entities, relations, classification, structured data, "
    "multi-task schemas, and batch processing.",
    version="0.3.0",
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


def _validators(specs: Optional[List[ValidatorSpec]]) -> List[RegexValidator]:
    return [RegexValidator(**spec) for spec in (specs or [])]


def _build_schema(model: GLiNER2, spec: Dict[str, Any]):
    """Build a gliner2 Schema from a plain JSON dict."""
    schema = model.create_schema()
    entities = spec.get("entities")
    if entities:
        schema = schema.entities(entities)
    for cls_task in spec.get("classifications", []):
        schema = schema.classification(
            cls_task["task"],
            cls_task["labels"],
            multi_label=cls_task.get("multi_label", False),
            cls_threshold=cls_task.get("cls_threshold", 0.5),
        )
    relations = spec.get("relations")
    if relations:
        schema = schema.relations(relations)
    for name, structure in spec.get("structures", {}).items():
        builder = schema.structure(name)
        for field_spec in structure.get("fields", []):
            if isinstance(field_spec, str):
                field_spec = {"name": field_spec}
            builder = builder.field(
                field_spec["name"],
                dtype=field_spec.get("dtype", "list"),
                choices=field_spec.get("choices"),
                description=field_spec.get("description"),
                threshold=field_spec.get("threshold"),
                validators=_validators(field_spec.get("validators")),
            )
    return schema


@serve.deployment
@serve.ingress(app)
class GLiNER2Deployment:
    def __init__(self):
        self.model = GLiNER2.from_pretrained(
            MODEL_ID,
            quantize=os.environ.get("GLINER_QUANTIZE", "").lower() in ("1", "true"),
            compile=os.environ.get("GLINER_COMPILE", "").lower() in ("1", "true"),
        )

    @app.post("/extract_entities", summary="Entity extraction")
    def extract_entities(self, request: ExtractEntitiesRequest):
        return self.model.extract_entities(
            request.text,
            request.entity_types,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/extract_relations", summary="Relation extraction")
    def extract_relations(self, request: ExtractRelationsRequest):
        return self.model.extract_relations(
            request.text,
            request.relation_types,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/classify_text", summary="Text classification (single/multi-label)")
    def classify_text(self, request: ClassifyTextRequest):
        return self.model.classify_text(
            request.text,
            request.tasks,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            max_len=request.max_len,
        )

    @app.post("/extract_json", summary="Structured data extraction")
    def extract_json(self, request: ExtractJsonRequest):
        return self.model.extract_json(
            request.text,
            request.structures,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/extract", summary="Multi-task schema extraction")
    def extract(self, request: ExtractRequest):
        schema = _build_schema(self.model, request.schema)
        return self.model.extract(
            request.text,
            schema,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/batch_extract_entities", summary="Batch entity extraction")
    def batch_extract_entities(self, request: BatchExtractEntitiesRequest):
        return self.model.batch_extract_entities(
            request.texts,
            request.entity_types,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/batch_extract_relations", summary="Batch relation extraction")
    def batch_extract_relations(self, request: BatchExtractRelationsRequest):
        return self.model.batch_extract_relations(
            request.texts,
            request.relation_types,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/batch_classify_text", summary="Batch text classification")
    def batch_classify_text(self, request: BatchClassifyTextRequest):
        return self.model.batch_classify_text(
            request.texts,
            request.tasks,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            max_len=request.max_len,
        )

    @app.post("/batch_extract_json", summary="Batch structured data extraction")
    def batch_extract_json(self, request: BatchExtractJsonRequest):
        return self.model.batch_extract_json(
            request.texts,
            request.structures,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )

    @app.post("/batch_extract", summary="Batch multi-task schema extraction")
    def batch_extract(self, request: BatchExtractRequest):
        schemas = request.schemas
        if isinstance(schemas, dict):
            schemas = _build_schema(self.model, schemas)
        else:
            schemas = [_build_schema(self.model, s) for s in schemas]
        return self.model.batch_extract(
            request.texts,
            schemas,
            batch_size=request.batch_size,
            threshold=request.threshold,
            include_confidence=request.include_confidence,
            include_spans=request.include_spans,
            max_len=request.max_len,
        )


app = GLiNER2Deployment.bind()  # ty: ignore[unresolved-attribute]  # added by @serve.deployment
