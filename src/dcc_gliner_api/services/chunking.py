"""Pure long-document chunking: split, batch, remap, merge. No model imports.

Splitting is word-aware using processor-compatible token rules so that
character offsets reported by the model align with this module's offsets.
Batching groups whole documents into windows of chunks, capped at
``batch_size``, so callers can make one model call per window.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from dcc_gliner_api.models.entities import Entity

CHUNK_SIZE = 384
CHUNK_OVERLAP = 64

EntityMap = dict[str, list[Entity]]

_WORD_PATTERN = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)


@dataclass(frozen=True)
class TextChunk:
    """One overlapping word window of a document, with offsets into the original text."""

    text: str
    start_char: int
    end_char: int
    start_word: int
    end_word: int


def iter_word_offsets(text: str) -> Iterable[tuple[str, int, int]]:
    """Yield (word, start_char, end_char) triples over processor-compatible tokens."""
    for match in _WORD_PATTERN.finditer(text):
        yield match.group(), match.start(), match.end()


def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split text into overlapping word windows (step = chunk_size - chunk_overlap)."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    tokens = list(iter_word_offsets(text))
    if not tokens:
        return [TextChunk(text=text, start_char=0, end_char=len(text), start_word=0, end_word=0)]

    chunks: list[TextChunk] = []
    step = chunk_size - chunk_overlap
    start_word = 0
    while start_word < len(tokens):
        end_word = min(start_word + chunk_size, len(tokens))
        start_char = tokens[start_word][1]
        end_char = tokens[end_word - 1][2]
        chunks.append(
            TextChunk(
                text=text[start_char:end_char],
                start_char=start_char,
                end_char=end_char,
                start_word=start_word,
                end_word=end_word,
            )
        )
        if end_word == len(tokens):
            break
        start_word += step
    return chunks


def iter_batch_windows(
    documents: Iterable[list[TextChunk]],
    batch_size: int,
) -> Iterator[list[list[TextChunk]]]:
    """Group documents' chunk lists into windows of at most ``batch_size`` chunks.

    A document's chunks are never split across windows: a document whose
    chunks alone exceed ``batch_size`` forms its own (oversized) window.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    window: list[list[TextChunk]] = []
    chunk_count = 0
    for chunks in documents:
        if window and chunk_count + len(chunks) > batch_size:
            yield window
            window = []
            chunk_count = 0
        window.append(chunks)
        chunk_count += len(chunks)
    if window:
        yield window


def remap_spans(entity_map: EntityMap, chunk: TextChunk) -> EntityMap:
    """Shift chunk-local character spans to global document offsets."""
    remapped: EntityMap = {}
    for label, items in entity_map.items():
        shifted: list[Entity] = []
        for item in items:
            global_item = remap_enity(item, chunk)
            shifted.append(global_item)
        remapped[label] = shifted
    return remapped


def remap_enity(entity: Entity, chunk: TextChunk) -> Entity:
    return Entity(
        start=chunk.start_char + entity.start,
        end=chunk.start_char + entity.end,
        confidence=entity.confidence,
        text=entity.text,
    )


def merge_detections(entity_map: EntityMap) -> EntityMap:
    """Merge per-chunk detections into one map.

    Per label, same-span overlapping detections collapse to the highest-confidence
    one (an overlap artifact); distinct mentions at different positions survive.
    Output is in document order.
    """
    labels: list[str] = []
    for label in entity_map:
        if label not in labels:
            labels.append(label)
    return {label: _merge_label(entity_map, label) for label in labels}


def _merge_label(entity_map: EntityMap, label: str) -> list[Entity]:
    items = entity_map.get(label, [])
    ranked = sorted(items, key=lambda i: (-i.confidence, i.start, i.end))
    selected: list[Entity] = []
    for item in ranked:
        if not any(_spans_overlap(item, chosen) for chosen in selected):
            selected.append(item)
    return sorted(selected, key=lambda i: (i.start, i.end))


def _spans_overlap(a: Entity, b: Entity) -> bool:
    return not (a.end <= b.start or a.start >= b.end)
