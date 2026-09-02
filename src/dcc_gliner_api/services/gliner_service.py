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
import threading
from collections import defaultdict
from collections.abc import Callable, Iterator
from typing import Any

import torch
from gliner2 import RegexValidator
from gliner2.inference.engine import GLiNER2
from gliner2.training.trainer import ExtractorCollator

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
from dcc_gliner_api.services.memory_plan import (
    DEFAULT_SAFETY_MARGIN,
    ActivationCost,
    derived_cost,
    plan_scan,
    profile_cost,
    split_labels,
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


def _activation_budget(device: str) -> int:
    """Activation memory a scan may use on this device.

    Read from the card the model actually landed on rather than configured per
    deployment: Ray hands the replica its own GPU, so what is free at startup
    is what this process has. ``GLINER_MEMORY_BUDGET_GIB`` overrides it, which
    is also how a CPU run gets a number at all.
    """
    configured = os.environ.get("GLINER_MEMORY_BUDGET_GIB")
    if configured:
        return int(float(configured) * 1024**3)

    if not device.startswith("cuda"):
        # Host memory is not the constraint a scan runs into; keep it generous.
        return 32 * 1024**3

    free, _total = torch.cuda.mem_get_info()
    return int(free * DEFAULT_SAFETY_MARGIN)


def _has_content(text: str) -> bool:
    """Whether a chunk holds anything a model could annotate."""
    return any(character.isalnum() for character in text)


class GlinerService:
    """Owns the GLiNER2 model lifecycle; the only code that touches the model."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        budget_bytes: int | None = None,
        sequence_sizer: Callable[[Any, str], int] | None = None,
    ):
        device = _device()
        logger.info("Loading GLiNER2 on %s", device)
        self.model: GLiNER2 = GLiNER2.from_pretrained(
            model_id or os.environ.get("GLINER_MODEL", DEFAULT_MODEL_ID),
            quantize=_env_flag("GLINER_QUANTIZE"),
            compile=_env_flag("GLINER_COMPILE"),
            map_location=device,
        )
        # One card, one scan: a second scan running beside this one would spend
        # the same memory twice and neither would know about the other.
        self._scanning = threading.Lock()
        self._budget_bytes = budget_bytes if budget_bytes is not None else _activation_budget(device)
        # Measuring a sequence needs the model's own tokenizer; taking it as a
        # collaborator lets a caller measure differently, or not at all.
        self._sequence_sizer = sequence_sizer or self._tokenized_length
        self._cost = profile_cost(self._weigh_one_chunk, self._derived_cost())
        logger.info(
            "Activation cost: %d bytes per token pair (%s)",
            self._cost.bytes_per_token_squared,
            self._cost.source,
        )
        logger.info("Activation budget: %.2f GiB", self._budget_bytes / 1024**3)

    def _derived_cost(self) -> ActivationCost:
        """What the model's own shape says a chunk costs."""
        config = self.model.encoder.config
        element_bytes = next(self.model.parameters()).element_size()
        return derived_cost(num_heads=config.num_attention_heads, bytes_per_element=element_bytes)

    def _weigh_one_chunk(self, words: int) -> tuple[int, int]:
        """Run one chunk of roughly ``words`` words and report what it cost."""
        if not torch.cuda.is_available() or _device() == "cpu":
            raise RuntimeError("no GPU to weigh")

        probe_text = " ".join(["Andreas Mueller wohnt in Muttenz."] * max(1, words // 5))
        entity_types = {"person": "Name einer Person", "ort": "Name einer Gemeinde"}
        sequence_length = self._tokenized_length(entity_types, probe_text)

        torch.cuda.synchronize()
        weights = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        self._scan_chunks([probe_text], entity_types, batch_size=1, threshold=0.5)
        torch.cuda.synchronize()

        return torch.cuda.max_memory_allocated() - weights, sequence_length

    def _tokenized_length(self, entity_types: Any, sample: str) -> int:
        """Tokens one chunk is scanned in, schema included."""
        schema = self.model.create_schema().entities(entity_types).build()
        collator = ExtractorCollator(self.model.processor, is_training=False, max_len=CHUNK_SIZE)
        return int(collator([(sample, schema)]).input_ids.shape[-1])

    def _scan_chunks(
        self,
        chunk_texts: list[str],
        entity_types: Any,
        *,
        batch_size: int,
        threshold: float,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """One model call, alone on the card."""
        with self._scanning:
            return self.model.batch_extract_entities(
                chunk_texts,
                entity_types,
                batch_size=batch_size,
                threshold=threshold,
                format_results=False,
                include_confidence=True,
                include_spans=True,
                max_len=CHUNK_SIZE,
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

    def _scan_groups(
        self,
        chunk_texts: list[str],
        groups: list[Any],
        *,
        batch_size: int,
        threshold: float,
    ) -> list[dict[str, list[dict[str, Any]]]]:
        """Scan one window of chunks, once per schema group.

        A schema too large to scan in one pass is scanned in pieces and read
        back together: the groups hold disjoint labels, so a chunk's findings
        are the union of what each group found in it.
        """
        # raw chunks returns: [{ entities: [ { person: [{text: str, ...}]} ]}]
        merged: list[dict[str, list[dict[str, Any]]]] = [{"entities": [{}]} for _ in chunk_texts]

        for group in groups:
            scanned = self._scan_chunks(chunk_texts, group, batch_size=batch_size, threshold=threshold)
            for chunk_entities, found in zip(merged, scanned, strict=True):
                chunk_entities["entities"][0].update(_entity_map_of(found))

        return merged

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

        sample = next((chunk.text for chunks in documents for chunk in chunks), "")
        sequence_length = self._sequence_sizer(entity_types, sample)
        plan = plan_scan(self._budget_bytes, sequence_length, batch_size, len(entity_types), self._cost)
        groups = split_labels(entity_types, plan.schema_groups)
        logger.info(
            "Scanning %d chunk(s): sequence %d tokens, batch %d, schema in %d group(s)",
            total_chunks,
            sequence_length,
            plan.batch_size,
            plan.schema_groups,
        )

        windows = iter_batch_windows(iter(documents), plan.batch_size)

        current_doc_index = 1
        for window in windows:
            raw_chunks = self._scan_groups(
                [chunk.text for chunks in window for chunk in chunks],
                groups,
                batch_size=plan.batch_size,
                threshold=threshold,
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
