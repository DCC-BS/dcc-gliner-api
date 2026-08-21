"""Service wrapping the GLiNER2 model: lifecycle, schema building, every model call.

Entity extraction runs as a chunk scan (see ``chunking``): the document is split
into overlapping chunks, each chunk is extracted with spans and confidence, spans
are remapped to global offsets, and overlap artifacts are merged away.

All other capabilities are thin delegates that keep ``format_results=False`` so
the upstream dedup formatter never drops repeated mentions (see ``formatting``).
"""

from __future__ import annotations
from collections import defaultdict
import debugpy

import os
from collections.abc import Iterator
from typing import Any

from gliner2 import RegexValidator
from gliner2.inference.engine import GLiNER2
from dcc_gliner_api.models.entities import Entity, ExtractEntitiesResponse

from dcc_gliner_api.services.chunking import (
    CHUNK_SIZE,
    EntityMap,
    iter_batch_windows,
    merge_detections,
    remap_spans,
    split_text_into_chunks, remap_enity,
)

DEFAULT_MODEL_ID = "fastino/gliner2-multi-v1"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true")


def _validators(specs: list[dict[str, Any]] | None) -> list[RegexValidator]:
    return [RegexValidator(**spec) for spec in (specs or [])]


def _entity_map_of(raw: dict[str, Any]) -> EntityMap:
    entities = raw.get("entities") or []
    return entities[0] if entities else {}


# def _project(
#     merged: EntityMap, include_confidence: bool, include_spans: bool
# ) -> dict[str, list[Entity]]:
#     projected: dict[str, Any] = {}
#     for label, items in merged.items():
#         if include_confidence and include_spans:
#             projected[label] = items
#         elif include_spans:
#             projected[label] = [
#                 {"text": i["text"], "start": i["start"], "end": i["end"]} for i in items
#             ]
#         elif include_confidence:
#             projected[label] = [
#                 {"text": i["text"], "confidence": i["confidence"]} for i in items
#             ]
#         else:
#             projected[label] = [i["text"] for i in items]
#     return projected


class GlinerService:
    """Owns the GLiNER2 model lifecycle; the only code that touches the model."""

    def __init__(self, model_id: str | None = None):
        self.model: GLiNER2 = GLiNER2.from_pretrained(
            model_id or os.environ.get("GLINER_MODEL", DEFAULT_MODEL_ID),
            quantize=_env_flag("GLINER_QUANTIZE"),
            compile=_env_flag("GLINER_COMPILE"),
        )

    def extract_entities(
        self,
        text: str,
        entity_types: Any,
        *,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> dict[str, Any]:
        return next(
            self.batch_extract_entities(
                [text],
                entity_types,
                threshold=threshold,
                include_confidence=include_confidence,
                include_spans=include_spans,
            )
        )

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: Any,
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> Iterator[ExtractEntitiesResponse]:
        """Lazily yield one merged entity result per document, in input order.

        Documents are chunked and grouped into windows of at most
        ``batch_size`` chunks (see ``chunking.iter_batch_windows``): each
        window is one model call, and its documents' results are yielded as
        soon as that window completes — ready for HTTP streaming. The chunk
        scan (split, remap, merge) runs per document inside the window.
        """
        windows = iter_batch_windows(
            (split_text_into_chunks(text) for text in texts), batch_size
        )

        # list[{ entities: [ { person: [{text: str, ...}]} ]}]

        for window in windows:
            raw_chunks: list[dict[str, list[dict[str, Any]]]] = self.model.batch_extract_entities(
                [chunk.text for chunks in window for chunk in chunks],
                entity_types,
                batch_size=batch_size,
                threshold=threshold,
                format_results=False,
                include_confidence=True,
                include_spans=True,
                max_len=CHUNK_SIZE,
            )

            offset = 0
            for chunks in window:
                document_chunks = raw_chunks[offset : offset + len(chunks)]
                offset += len(chunks)

                entities: dict[str, list[Entity]] = defaultdict(list, [])
                for chunk, raw in zip(chunks, document_chunks):
                    for label, entity_list in raw["entities"][0].items():
                        for entity_map in entity_list:
                            relative_enity = Entity.model_validate(entity_map)
                            entities[label].append(remap_enity(relative_enity, chunk))

                yield ExtractEntitiesResponse(entities=merge_detections(entities))


    def build_schema(self, spec: dict[str, Any]):
        """Build a gliner2 Schema from a plain JSON dict."""
        schema = self.model.create_schema()
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

    def extract_relations(
        self,
        text: str,
        relation_types: Any,
        *,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any]:
        return self.model.extract_relations(
            text,
            relation_types,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )

    def batch_extract_relations(
        self,
        texts: list[str],
        relation_types: Any,
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.model.batch_extract_relations(
            texts,
            relation_types,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )

    def classify_text(
        self,
        text: str,
        tasks: dict[str, Any],
        *,
        threshold: float = 0.5,
        include_confidence: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any]:
        return self.model.classify_text(
            text,
            tasks,
            threshold=threshold,
            include_confidence=include_confidence,
            max_len=max_len,
            format_results=False,
        )

    def batch_classify_text(
        self,
        texts: list[str],
        tasks: dict[str, Any],
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.model.batch_classify_text(
            texts,
            tasks,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=include_confidence,
            max_len=max_len,
            format_results=False,
        )

    def extract_json(
        self,
        text: str,
        structures: dict[str, Any],
        *,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any]:
        return self.model.extract_json(
            text,
            structures,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )

    def batch_extract_json(
        self,
        texts: list[str],
        structures: dict[str, Any],
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.model.batch_extract_json(
            texts,
            structures,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )

    def extract(
        self,
        text: str,
        schema: Any,
        *,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> dict[str, Any]:
        return self.model.extract(
            text,
            schema,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )

    def batch_extract(
        self,
        texts: list[str],
        schemas: Any,
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
        max_len: int | None = None,
    ) -> list[dict[str, Any]]:
        return self.model.batch_extract(
            texts,
            schemas,
            batch_size=batch_size,
            threshold=threshold,
            include_confidence=include_confidence,
            include_spans=include_spans,
            max_len=max_len,
            format_results=False,
        )
