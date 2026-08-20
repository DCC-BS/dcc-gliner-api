"""Pure long-document chunking: split, remap, merge. No model imports.

Splitting is word-aware using processor-compatible token rules so that
character offsets reported by the model align with this module's offsets.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

CHUNK_SIZE = 384
CHUNK_OVERLAP = 64

SpanItem = dict[str, Any]
EntityMap = dict[str, list[SpanItem]]

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
        return [
            TextChunk(
                text=text, start_char=0, end_char=len(text), start_word=0, end_word=0
            )
        ]

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


def remap_spans(entity_map: EntityMap, chunk: TextChunk) -> EntityMap:
    """Shift chunk-local character spans to global document offsets."""
    remapped: EntityMap = {}
    for label, items in entity_map.items():
        shifted = []
        for item in items:
            global_item = dict(item)
            global_item["start"] += chunk.start_char
            global_item["end"] += chunk.start_char
            shifted.append(global_item)
        remapped[label] = shifted
    return remapped


def merge_detections(per_chunk_maps: list[EntityMap]) -> EntityMap:
    """Merge per-chunk detections into one map.

    Per label, same-span overlapping detections collapse to the highest-confidence
    one (an overlap artifact); distinct mentions at different positions survive.
    Output is in document order.
    """
    labels: list[str] = []
    for entity_map in per_chunk_maps:
        for label in entity_map:
            if label not in labels:
                labels.append(label)
    return {label: _merge_label(per_chunk_maps, label) for label in labels}


def _merge_label(per_chunk_maps: list[EntityMap], label: str) -> list[SpanItem]:
    items = [item for m in per_chunk_maps for item in m.get(label, [])]
    ranked = sorted(
        items, key=lambda i: (-i.get("confidence", 0.0), i["start"], i["end"])
    )
    selected: list[SpanItem] = []
    for item in ranked:
        if not any(_spans_overlap(item, chosen) for chosen in selected):
            selected.append(item)
    return sorted(selected, key=lambda i: (i["start"], i["end"]))


def _spans_overlap(a: SpanItem, b: SpanItem) -> bool:
    return not (a["end"] <= b["start"] or a["start"] >= b["end"])
