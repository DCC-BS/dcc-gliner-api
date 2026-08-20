"""Unit tests for GlinerService.iter_batch_extract_entities with a stubbed model.

The stub records model calls and returns each chunk's first word as a probe
mention with chunk-local offsets, exercising the real split -> model call ->
remap -> merge -> project path without loading torch weights.
"""

import pytest

from dcc_gliner_api.services.gliner_service import GlinerService


class StubModel:
    def __init__(self):
        self.calls: list[list[str]] = []

    def batch_extract_entities(self, chunk_texts, entity_types, **kwargs):
        self.calls.append(list(chunk_texts))
        results = []
        for chunk_text in chunk_texts:
            words = chunk_text.split()
            first = words[0] if words else ""
            start = chunk_text.find(first)
            results.append(
                {
                    "entities": [
                        {
                            "probe": [
                                {
                                    "text": first,
                                    "confidence": 0.9,
                                    "start": start,
                                    "end": start + len(first),
                                }
                            ]
                        }
                    ]
                }
            )
        return results


@pytest.fixture
def service():
    svc = GlinerService.__new__(GlinerService)
    svc.model = StubModel()
    return svc


def test_one_result_per_document_in_order(service):
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    results = list(service.iter_batch_extract_entities(texts, ["probe"]))
    assert [r["probe"][0] for r in results] == ["alpha", "gamma", "epsilon"]


def test_windows_respect_batch_size(service):
    texts = [f"doc{i} body" for i in range(5)]
    list(service.iter_batch_extract_entities(texts, ["probe"], batch_size=2))
    assert [len(call) for call in service.model.calls] == [2, 2, 1]


def test_document_chunks_never_split_across_model_calls(service):
    text = " ".join(f"w{i}" for i in range(1000))
    list(
        service.iter_batch_extract_entities(
            [text, "short doc"], ["probe"], batch_size=2
        )
    )
    assert [len(call) for call in service.model.calls] == [3, 1]


def test_generator_is_lazy(service):
    texts = [f"doc{i} body" for i in range(5)]
    it = service.iter_batch_extract_entities(texts, ["probe"], batch_size=2)
    next(it)
    assert len(service.model.calls) == 1
    next(it)
    assert len(service.model.calls) == 1
    next(it)
    assert len(service.model.calls) == 2
    assert len(list(it)) == 2
    assert len(service.model.calls) == 3


def test_empty_texts_yield_nothing(service):
    assert list(service.iter_batch_extract_entities([], ["probe"])) == []
    assert service.model.calls == []


def test_global_span_invariant_through_stub(service):
    text = " ".join(f"w{i}" for i in range(1000))
    (result,) = service.batch_extract_entities([text], ["probe"], include_spans=True)
    assert len(result["probe"]) > 1
    for item in result["probe"]:
        assert text[item["start"] : item["end"]] == item["text"]


def test_projection_strips_unrequested_fields(service):
    (result,) = service.batch_extract_entities(
        ["alpha beta"], ["probe"], include_spans=True
    )
    (item,) = result["probe"]
    assert set(item) == {"text", "start", "end"}
    (result,) = service.batch_extract_entities(
        ["alpha beta"], ["probe"], include_confidence=True, include_spans=True
    )
    (item,) = result["probe"]
    assert set(item) == {"text", "confidence", "start", "end"}
