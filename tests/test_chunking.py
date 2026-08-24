"""Unit tests for the pure chunking module: split, remap, merge.

No model or torch imports required — chunking is model-free by design.
"""

import re
from itertools import pairwise

import pytest

from dcc_gliner_api.models.entities import Entity
from dcc_gliner_api.services.chunking import (
    CHUNK_SIZE,
    TextChunk,
    iter_batch_windows,
    merge_detections,
    remap_spans,
    split_text_into_chunks,
)


def mention(text, confidence, start, end):
    return Entity(text=text, confidence=confidence, start=start, end=end)


def find_word(chunk_text, needle):
    match = re.search(rf"\b{re.escape(needle)}\b", chunk_text)
    return match.start() if match else -1


def make_document(word_count):
    return " ".join(f"w{i}" for i in range(word_count))


@pytest.fixture
def long_document():
    return make_document(1000)


def make_doc_chunks(count, tag="d"):
    return [
        TextChunk(
            text=f"{tag}{i}",
            start_char=i,
            end_char=i + 1,
            start_word=i,
            end_word=i + 1,
        )
        for i in range(count)
    ]


class TestSplit:
    def test_offsets_roundtrip(self, long_document):
        for chunk in split_text_into_chunks(long_document):
            assert long_document[chunk.start_char : chunk.end_char] == chunk.text

    def test_full_word_coverage(self, long_document):
        chunks = split_text_into_chunks(long_document)
        covered = set()
        for chunk in chunks:
            covered.update(range(chunk.start_word, chunk.end_word))
        assert covered == set(range(1000))

    def test_adjacent_chunks_overlap(self, long_document):
        chunks = split_text_into_chunks(long_document)
        for left, right in pairwise(chunks):
            assert right.start_word < left.end_word
            assert right.end_word > left.end_word

    def test_overlap_is_chunk_overlap_words_when_unclipped(self):
        text = make_document(500)
        chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)
        for left, right in pairwise(chunks):
            if right.end_word < 500:
                assert left.end_word - right.start_word == 20

    def test_chunks_never_exceed_chunk_size(self, long_document):
        for chunk in split_text_into_chunks(long_document):
            assert chunk.end_word - chunk.start_word <= CHUNK_SIZE

    def test_short_text_is_one_chunk(self):
        text = "Apple CEO Tim Cook announced iPhone 15 in Cupertino."
        chunks = split_text_into_chunks(text)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert (chunks[0].start_char, chunks[0].end_char) == (0, len(text))

    def test_empty_text_is_one_chunk(self):
        chunks = split_text_into_chunks("")
        assert chunks == [TextChunk(text="", start_char=0, end_char=0, start_word=0, end_word=0)]

    def test_invalid_params_raise(self):
        with pytest.raises(ValueError):
            split_text_into_chunks("text", chunk_size=0)
        with pytest.raises(ValueError):
            split_text_into_chunks("text", chunk_overlap=-1)
        with pytest.raises(ValueError):
            split_text_into_chunks("text", chunk_size=10, chunk_overlap=10)


class TestBatchWindows:
    def test_groups_documents_up_to_batch_size(self):
        docs = [make_doc_chunks(n, tag=f"d{i}") for i, n in enumerate((2, 2, 2, 3))]
        windows = list(iter_batch_windows(docs, batch_size=4))
        totals = [sum(len(chunks) for chunks in window) for window in windows]
        assert totals == [4, 2, 3]

    def test_documents_stay_intact_and_in_order(self):
        docs = [make_doc_chunks(n, tag=f"d{i}") for i, n in enumerate((1, 2, 1, 3))]
        windows = list(iter_batch_windows(docs, batch_size=3))
        flattened = [chunks for window in windows for chunks in window]
        assert flattened == docs

    def test_oversized_document_gets_own_window(self):
        docs = [make_doc_chunks(n, tag=f"d{i}") for i, n in enumerate((3, 10, 2))]
        windows = list(iter_batch_windows(docs, batch_size=4))
        totals = [sum(len(chunks) for chunks in window) for window in windows]
        assert totals == [3, 10, 2]
        for window, total in zip(windows, totals, strict=True):
            if total > 4:
                assert len(window) == 1

    def test_windows_from_real_split_cover_every_chunk(self, long_document):
        docs = [
            split_text_into_chunks(long_document, chunk_size=100, chunk_overlap=20),
            split_text_into_chunks("short text"),
        ]
        windows = list(iter_batch_windows(docs, batch_size=8))
        flattened = [chunk for window in windows for chunks in window for chunk in chunks]
        assert flattened == [chunk for chunks in docs for chunk in chunks]

    def test_empty_input_yields_nothing(self):
        assert list(iter_batch_windows([], batch_size=4)) == []

    def test_invalid_batch_size_raises(self):
        with pytest.raises(ValueError):
            list(iter_batch_windows([], batch_size=0))


class TestRemap:
    def test_local_spans_shift_to_global(self):
        document = " ".join(f"w{i}" for i in range(100))
        chunks = split_text_into_chunks(document, chunk_size=20, chunk_overlap=4)
        chunk = chunks[1]
        local = chunk.text.find("w25")
        assert local >= 0
        entity_map = {"probe": [mention("w25", 0.9, local, local + len("w25"))]}
        global_map = remap_spans(entity_map, chunk)
        start, end = global_map["probe"][0].start, global_map["probe"][0].end
        assert document[start:end] == "w25"

    def test_empty_map_passes_through(self):
        chunk = split_text_into_chunks("hello world")[0]
        assert remap_spans({}, chunk) == {}


class TestMerge:
    def test_overlap_artifact_collapses_to_higher_confidence(self):
        low = mention("Tim Cook", 0.70, 100, 108)
        high = mention("Tim Cook", 0.95, 101, 109)
        merged = merge_detections({"person": [low, high]})
        assert merged["person"] == [high]

    def test_distinct_mentions_survive(self):
        first = mention("Apple", 0.9, 10, 15)
        second = mention("Apple", 0.8, 500, 505)
        merged = merge_detections({"company": [first, second]})
        assert merged["company"] == [first, second]

    def test_overlap_collapses_to_higher_confidence_regardless_of_length(self):
        fragment = mention("Cook", 0.99, 104, 108)
        full = mention("Tim Cook", 0.80, 100, 108)
        assert merge_detections({"person": [fragment, full]})["person"] == [fragment]
        assert merge_detections({"person": [full, fragment]})["person"] == [fragment]
        weaker_fragment = mention("Cook", 0.70, 104, 108)
        assert merge_detections({"person": [weaker_fragment, full]})["person"] == [full]

    def test_identical_span_keeps_best_confidence(self):
        a = mention("Apple", 0.8, 10, 15)
        b = mention("Apple", 0.95, 10, 15)
        merged = merge_detections({"company": [a, b]})
        assert merged["company"] == [b]

    def test_output_in_document_order(self):
        late = mention("zeta", 0.99, 900, 904)
        early = mention("alpha", 0.51, 100, 105)
        mid = mention("beta", 0.7, 500, 504)
        merged = merge_detections({"label": [late, early, mid]})
        assert merged["label"] == [early, mid, late]

    def test_labels_merge_independently(self):
        person = mention("Cook", 0.9, 100, 104)
        role = mention("Cook", 0.8, 100, 104)
        merged = merge_detections({"person": [person], "role": [role]})
        assert merged["person"] == [person]
        assert merged["role"] == [role]

    def test_empty_inputs(self):
        assert merge_detections({}) == {}


class TestPipelineInvariant:
    def test_mention_found_in_every_chunk_maps_to_document(self):
        document = " ".join(f"w{i}" for i in range(800))
        chunks = split_text_into_chunks(document, chunk_size=64, chunk_overlap=16)
        combined: dict[str, list[Entity]] = {}
        for chunk in chunks:
            local = find_word(chunk.text, "w50")
            if local >= 0:
                remapped = remap_spans(
                    {"probe": [mention("w50", 0.9, local, local + len("w50"))]},
                    chunk,
                )
                combined.setdefault("probe", []).extend(remapped["probe"])
        assert len(combined["probe"]) > 1
        merged = merge_detections(combined)
        assert len(merged["probe"]) == 1
        item = merged["probe"][0]
        assert document[item.start : item.end] == "w50"

    def test_repeated_mentions_at_distinct_positions_survive(self):
        document = " ".join(f"w{i}" for i in range(400))
        chunks = split_text_into_chunks(document, chunk_size=50, chunk_overlap=10)
        combined: dict[str, list[Entity]] = {}
        for chunk in chunks:
            entity_map = {}
            for needle in ("w5", "w300"):
                local = find_word(chunk.text, needle)
                if local >= 0:
                    entity_map.setdefault("probe", []).append(mention(needle, 0.9, local, local + len(needle)))
            remapped = remap_spans(entity_map, chunk)
            if "probe" in remapped:
                combined.setdefault("probe", []).extend(remapped["probe"])
        merged = merge_detections(combined)
        texts = [item.text for item in merged["probe"]]
        assert texts == ["w5", "w300"]
        for item in merged["probe"]:
            assert document[item.start : item.end] == item.text
