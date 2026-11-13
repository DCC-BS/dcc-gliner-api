"""Service wrapping the GLiNER2 model: lifecycle, schema building, every model call.

Entity extraction runs as a chunk scan (see ``chunking``): the document is split
into overlapping chunks, each chunk is extracted with spans and confidence, spans
are remapped to global offsets, and overlap artifacts are merged away.

Every chunk is measured before it is scanned, and each model call is sized for
the longest chunk it carries (see ``memory_plan``). Sizing a whole document by
its first chunk is how a cover page of prose followed by pages of links or
encoded data asked for many times the memory the card had.

All other capabilities are thin delegates that keep ``format_results=False`` so
the upstream dedup formatter never drops repeated mentions (see ``formatting``).
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import torch
from gliner2 import RegexValidator
from gliner2.inference.engine import GLiNER2
from gliner2.training.trainer import ExtractorCollator

from dcc_gliner_api.models.common import BatchProgress
from dcc_gliner_api.models.entities import Entity, ExtractEntitiesBatchResponse, ExtractEntitiesResponse
from dcc_gliner_api.services.chunking import (
    CHUNK_SIZE,
    MAX_CHUNK_TOKENS,
    EntityMap,
    TextChunk,
    bound_chunk_tokens,
    count_text_tokens,
    iter_batch_windows,
    merge_detections,
    remap_enity,
    split_text_into_chunks,
)
from dcc_gliner_api.services.memory_plan import (
    DEFAULT_SAFETY_MARGIN,
    ActivationCost,
    ScanPlan,
    derived_cost,
    plan_scan,
    profile_cost,
    split_labels,
)

logger = logging.getLogger("ray.serve")

DEFAULT_MODEL_ID = "fastino/gliner2-multi-v1"

#: Chunks one model call takes, unless the memory plan reduces it.
DEFAULT_BATCH_SIZE = 8


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


def _gather_entities(
    entities: dict[str, list[Entity]],
    chunks: Iterable[TextChunk],
    raw_chunks: list[dict[str, Any]],
    entity_types: Any,
) -> None:
    """Read a scan's chunks into ``entities``, in chunk order.

    A chunk with no words (blank page, table rule) has nothing to annotate.
    One with words that annotates to nothing has not been read, and losing
    that silently is how a document comes back under-redacted, so it warns.
    """
    for chunk, raw in zip(chunks, raw_chunks, strict=True):
        entity_map = _entity_map_of(raw)
        if not entity_map:
            if _has_content(chunk.text):
                logger.warning(
                    "Chunk annotated to nothing: %d chars, %d label(s)",
                    len(chunk.text),
                    len(entity_types),
                )
            continue
        for label, entity_list in entity_map.items():
            for raw_entity in entity_list:
                relative_entity = Entity.model_validate(raw_entity)
                entities[label].append(remap_enity(relative_entity, chunk))


class GlinerService:
    """Owns the GLiNER2 model lifecycle; the only code that touches the model."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        budget_bytes: int | None = None,
        schema_sizer: Callable[[Any], int] | None = None,
        word_tokens: Callable[[str], int] | None = None,
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
        # Measuring a sequence needs the model's own tokenizer; taking both
        # halves as collaborators lets a caller measure differently.
        self._schema_sizer = schema_sizer or self._schema_length
        self._word_tokens = word_tokens or self._subword_counter()
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

    def _subword_counter(self) -> Callable[[str], int]:
        """Subword tokens one word is read in, as the processor reads it.

        The processor lower-cases and tokenises word by word, so a document's
        length in tokens is the sum over its words. Words repeat, so it is
        cached.
        """
        tokenize = self.model.processor.tokenizer.tokenize

        @functools.lru_cache(maxsize=65536)
        def count(word: str) -> int:
            return len(tokenize(word.lower()))

        return count

    def _schema_length(self, entity_types: Any) -> int:
        """Tokens the schema adds to every chunk, separators included."""
        return self._tokenized_length(entity_types, "")

    def _chunks_of(self, text: str) -> list[TextChunk]:
        """The document's chunks, none reading to more than the bound."""
        return bound_chunk_tokens(text, split_text_into_chunks(text), self._word_tokens, MAX_CHUNK_TOKENS)

    def _sequence_lengths(self, entity_types: Any, chunks: Iterable[TextChunk]) -> list[int]:
        """Tokens each chunk is scanned in, schema included."""
        schema_length = self._schema_sizer(entity_types)
        return [schema_length + count_text_tokens(chunk.text, self._word_tokens) for chunk in chunks]

    def _plan(self, sequence_lengths: list[int], wanted_batch: int, label_count: int) -> ScanPlan:
        """How to scan chunks of these lengths together: sized for the longest."""
        return plan_scan(self._budget_bytes, max(sequence_lengths), wanted_batch, label_count, self._cost)

    def _log_scan(self, sequence_lengths: list[int], plan: ScanPlan) -> None:
        logger.info(
            "Scanning %d chunk(s): sequence %d-%d tokens, batch %d, schema in %d group(s) for the longest",
            len(sequence_lengths),
            min(sequence_lengths),
            max(sequence_lengths),
            plan.batch_size,
            plan.schema_groups,
        )

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
        """Scan one text, and say where it stands between model calls.

        The batch scan only reports progress as it is driven past a yield, so
        the single-document call — one ``next()`` — heard it exactly once, at
        zero, and kept it there for however long the scan took. Here the
        document's chunks are scanned in slices, and each slice's count is
        passed on as it is done, so a long scan reads as a moving fraction
        rather than a stuck zero.

        Each slice is as many chunks as fit beside its longest one, so a run
        of long chunks deep in the document is scanned in smaller calls
        rather than in a batch sized for the prose at its start.
        """
        chunks = self._chunks_of(text)
        total_chunks = len(chunks)
        if on_progress:
            on_progress(0, total_chunks)

        lengths = self._sequence_lengths(entity_types, chunks)
        self._log_scan(lengths, self._plan(lengths, DEFAULT_BATCH_SIZE, len(entity_types)))

        entities: dict[str, list[Entity]] = defaultdict(list, [])
        done_chunks = 0
        start = 0
        while start < total_chunks:
            plan = self._plan(lengths[start : start + DEFAULT_BATCH_SIZE], DEFAULT_BATCH_SIZE, len(entity_types))
            # The plan may only shrink as the slice does: a slice cut to the
            # planned size holds no chunk longer than the one it was sized for.
            slice_chunks = chunks[start : start + plan.batch_size]
            raw_chunks = self._scan_groups(
                [chunk.text for chunk in slice_chunks],
                split_labels(entity_types, plan.schema_groups),
                batch_size=plan.batch_size,
                threshold=threshold,
            )
            start += len(slice_chunks)
            _gather_entities(entities, slice_chunks, raw_chunks, entity_types)

            done_chunks += len(slice_chunks)
            if on_progress:
                on_progress(done_chunks, total_chunks)

        return ExtractEntitiesResponse(entities=merge_detections(entities))

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
        documents = [self._chunks_of(text) for text in texts]
        total_chunks = sum(len(chunks) for chunks in documents)
        done_chunks = 0
        if on_progress:
            on_progress(0, total_chunks)

        if not documents:
            return

        # The whole batch is planned for its longest chunk, so windows are cut
        # to a size every one of them can be scanned at.
        lengths = self._sequence_lengths(entity_types, (chunk for chunks in documents for chunk in chunks))
        plan = self._plan(lengths, batch_size, len(entity_types))
        groups = split_labels(entity_types, plan.schema_groups)
        self._log_scan(lengths, plan)

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
                _gather_entities(entities, chunks, document_chunks, entity_types)

                progress = BatchProgress.new(current_doc_index, length=len(texts))
                current_doc_index += 1
                yield ExtractEntitiesBatchResponse(entities=merge_detections(entities), progress=progress)

            done_chunks += sum(len(chunks) for chunks in window)
            if on_progress:
                on_progress(done_chunks, total_chunks)
