"""Service wrapping the GLiNER2 model: lifecycle, schema building, every model call.

Entity extraction runs as a chunk scan (see ``chunking``): the document is split
into overlapping chunks, each chunk is extracted with spans and confidence, spans
are remapped to global offsets, and overlap artifacts are merged away.

All other capabilities are thin delegates that keep ``format_results=False`` so
the upstream dedup formatter never drops repeated mentions (see ``formatting``).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import torch
from gliner2 import RegexValidator
from gliner2.inference.engine import GLiNER2

from dcc_gliner_api.models.common import BatchProgress
from dcc_gliner_api.models.entities import Entity, ExtractEntitiesBatchResponse, ExtractEntitiesResponse
from dcc_gliner_api.services.chunking import (
    CHUNK_SIZE,
    EntityMap,
    iter_batch_windows,
    merge_detections,
    remap_enity,
    split_text_into_chunks,
)

logger = logging.getLogger("ray.serve")

DEFAULT_MODEL_ID = "fastino/gliner2-multi-v1"


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in ("1", "true")


def _device() -> str:
    """Pick where the model runs, preferring a GPU when one is present.

    GLiNER2 loads its weights with ``map_location="cpu"`` unless told
    otherwise, so without this the model runs on CPU even on a GPU node —
    orders of magnitude slower. Set ``GLINER_DEVICE`` to override.
    """
    configured = os.environ.get("GLINER_DEVICE")
    if configured:
        return configured

    return "cuda" if torch.cuda.is_available() else "cpu"


def _validators(specs: list[dict[str, Any]] | None) -> list[RegexValidator]:
    return [RegexValidator(**spec) for spec in (specs or [])]


def _entity_map_of(raw: dict[str, Any]) -> EntityMap:
    entities = raw.get("entities") or []
    return entities[0] if entities else {}


def _has_content(text: str) -> bool:
    """Whether a chunk holds anything a model could annotate."""
    return any(character.isalnum() for character in text)


class GlinerService:
    """Owns the GLiNER2 model lifecycle; the only code that touches the model."""

    def __init__(self, model_id: str | None = None):
        device = _device()
        logger.info("Loading GLiNER2 on %s", device)
        self.model: GLiNER2 = GLiNER2.from_pretrained(
            model_id or os.environ.get("GLINER_MODEL", DEFAULT_MODEL_ID),
            quantize=_env_flag("GLINER_QUANTIZE"),
            compile=_env_flag("GLINER_COMPILE"),
            map_location=device,
        )

    def extract_entities(
        self,
        text: str,
        entity_types: Any,
        *,
        threshold: float = 0.5,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> ExtractEntitiesResponse:
        return next(
            self.batch_extract_entities(
                [text],
                entity_types,
                threshold=threshold,
                on_progress=on_progress,
            )
        )

    def batch_extract_entities(
        self,
        texts: list[str],
        entity_types: Any,
        *,
        batch_size: int = 8,
        threshold: float = 0.5,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> Iterator[ExtractEntitiesBatchResponse]:
        """Lazily yield one merged entity result per document, in input order.

        Documents are chunked and grouped into windows of at most
        ``batch_size`` chunks (see ``chunking.iter_batch_windows``): each
        window is one model call, and its documents' results are yielded as
        soon as that window completes — ready for HTTP streaming. The chunk
        scan (split, remap, merge) runs per document inside the window.
        """
        # Chunks are the unit of work, so counting them up front lets a caller
        # watch a single long document progress instead of waiting on one yield.
        documents = [split_text_into_chunks(text) for text in texts]
        total_chunks = sum(len(chunks) for chunks in documents)
        done_chunks = 0
        if on_progress:
            on_progress(0, total_chunks)

        windows = iter_batch_windows(iter(documents), batch_size)

        current_doc_index = 1
        for window in windows:
            # raw chunks returns: [{ entities: [ { person: [{text: str, ...}]} ]}]
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
                for chunk, raw in zip(chunks, document_chunks, strict=True):
                    entity_map = _entity_map_of(raw)
                    if not entity_map:
                        # A chunk with no words (blank page, table rule) has
                        # nothing to annotate. One with words that annotates to
                        # nothing has not been read, and losing that silently is
                        # how a document comes back under-redacted.
                        if _has_content(chunk.text):
                            logger.warning(
                                "Chunk annotated to nothing: %d chars, %d label(s)",
                                len(chunk.text),
                                len(entity_types),
                            )
                        continue
                    for label, entity_list in entity_map.items():
                        for raw_entity in entity_list:
                            relative_enity = Entity.model_validate(raw_entity)
                            entities[label].append(remap_enity(relative_enity, chunk))

                progress = BatchProgress.new(current_doc_index, length=len(texts))
                current_doc_index += 1
                yield ExtractEntitiesBatchResponse(entities=merge_detections(entities), progress=progress)

            done_chunks += sum(len(chunks) for chunks in window)
            if on_progress:
                on_progress(done_chunks, total_chunks)
