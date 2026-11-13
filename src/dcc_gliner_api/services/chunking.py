"""Pure long-document chunking: split, batch, remap, merge. No model imports.

Splitting is word-aware using processor-compatible token rules so that
character offsets reported by the model align with this module's offsets.
Chunks are then bounded in subword tokens too (``bound_chunk_tokens``), given
a caller-supplied count per word, so that no chunk reads to more than the
scan's memory plan allows for.
Batching groups whole documents into windows of chunks, capped at
``batch_size``, so callers can make one model call per window.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from dcc_gliner_api.models.entities import Entity

CHUNK_SIZE = 384
CHUNK_OVERLAP = 64

#: Most subword tokens one chunk's text may read to. Prose fills a 384-word
#: chunk with roughly 750-950 tokens, so it is never cut; only text whose
#: words are long runs of subwords (links, blobs, hashes, glued words) is.
MAX_CHUNK_TOKENS = 1024

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


def count_text_tokens(text: str, word_tokens: Callable[[str], int]) -> int:
    """Subword tokens ``text`` is read in, given what each word costs."""
    return sum(word_tokens(word) for word, _, _ in iter_word_offsets(text))


def bound_chunk_tokens(
    text: str,
    chunks: list[TextChunk],
    word_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_words: int = CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Cut every chunk that reads to more than ``max_tokens`` subword tokens.

    Chunks are cut by words, but the encoder reads subword tokens, and one
    word can be hundreds of them: a link, a base64 blob, a hash, text whose
    spaces were lost in extraction. Attention memory grows with the square of
    what the encoder reads, so a chunk of such words costs many times what a
    chunk of prose does. Chunks within the bound are returned unchanged.

    An oversized chunk is re-cut into word windows that fit, overlapping by up
    to ``overlap_words`` words (and at most a quarter of the bound). A single
    word over the bound is cut at character boundaries into pieces of its own,
    each of which fits.

    Args:
        text: The document the chunks index into.
        chunks: The document's word chunks, in order.
        word_tokens: Subword tokens one word is read in.
        max_tokens: Most subword tokens a chunk may read to.
        overlap_words: Most words two re-cut neighbours share.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")

    bounded: list[TextChunk] = []
    for chunk in chunks:
        words = [
            (chunk.start_char + start, chunk.start_char + end, chunk.start_word + index, word_tokens(word))
            for index, (word, start, end) in enumerate(iter_word_offsets(chunk.text))
        ]
        if sum(cost for *_, cost in words) <= max_tokens:
            bounded.append(chunk)
        else:
            bounded.extend(_recut(text, words, word_tokens, max_tokens, overlap_words))
    return bounded


_Word = tuple[int, int, int, int]  # start_char, end_char, word index, subword tokens


def _recut(
    text: str,
    words: list[_Word],
    word_tokens: Callable[[str], int],
    max_tokens: int,
    overlap_words: int,
) -> Iterator[TextChunk]:
    """Re-cut one oversized chunk's words into windows within ``max_tokens``."""
    run: list[_Word] = []
    for word in words:
        if word[3] > max_tokens:
            yield from _windows(text, run, max_tokens, overlap_words)
            run = []
            yield from _split_word(text, word, word_tokens, max_tokens)
        else:
            run.append(word)
    yield from _windows(text, run, max_tokens, overlap_words)


def _windows(text: str, words: list[_Word], max_tokens: int, overlap_words: int) -> Iterator[TextChunk]:
    """Overlapping windows over ``words``, none above ``max_tokens``."""
    start = 0
    while start < len(words):
        end, used = start, 0
        while end < len(words) and used + words[end][3] <= max_tokens:
            used += words[end][3]
            end += 1
        yield _chunk_of(text, words[start][0], words[end - 1][1], words[start][2], words[end - 1][2] + 1)
        if end == len(words):
            return

        # Step back into the window for overlap, but always move forward.
        back, shared = 0, 0
        while back < overlap_words and end - back - 1 > start:
            cost = words[end - back - 1][3]
            if shared + cost > max_tokens // 4:
                break
            shared += cost
            back += 1
        start = end - back


def _split_word(text: str, word: _Word, word_tokens: Callable[[str], int], max_tokens: int) -> Iterator[TextChunk]:
    """Cut one word that alone is over the bound into pieces that are not."""
    start_char, end_char, index, cost = word
    # Characters per piece, from the word's own density, halved until it fits.
    width = max(1, (end_char - start_char) * max_tokens // cost)
    position = start_char
    while position < end_char:
        end = min(end_char, position + width)
        while end - position > 1 and count_text_tokens(text[position:end], word_tokens) > max_tokens:
            end = position + (end - position) // 2
        yield _chunk_of(text, position, end, index, index + 1)
        position = end


def _chunk_of(text: str, start_char: int, end_char: int, start_word: int, end_word: int) -> TextChunk:
    return TextChunk(
        text=text[start_char:end_char],
        start_char=start_char,
        end_char=end_char,
        start_word=start_word,
        end_word=end_word,
    )


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
