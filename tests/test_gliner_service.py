"""Unit tests for GlinerService.batch_extract_entities with a stubbed model.

The stub records model calls and returns each chunk's first word as a probe
mention with chunk-local offsets, exercising the real split -> model call ->
remap -> merge path without loading torch weights.
"""

import threading

import pytest

from dcc_gliner_api.services.gliner_service import GlinerService
from dcc_gliner_api.services.memory_plan import ActivationCost


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
            results.append({
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
            })
        return results


@pytest.fixture
def service():
    """A service with a stubbed model, a roomy budget and a fixed sequence."""
    svc = GlinerService.__new__(GlinerService)
    svc.model = StubModel()
    svc._scanning = threading.Lock()
    svc._budget_bytes = 32 * 1024**3
    svc._sequence_sizer = lambda entity_types, sample: 512
    svc._cost = ActivationCost(bytes_per_token_squared=360, source="test")
    return svc


def test_one_result_per_document_in_order(service):
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    results = list(service.batch_extract_entities(texts, ["probe"]))
    assert [r.entities["probe"][0].text for r in results] == [
        "alpha",
        "gamma",
        "epsilon",
    ]


def test_windows_respect_batch_size(service):
    texts = [f"doc{i} body" for i in range(5)]
    list(service.batch_extract_entities(texts, ["probe"], batch_size=2))
    assert [len(call) for call in service.model.calls] == [2, 2, 1]


def test_model_calls_never_exceed_batch_size(service):
    text = " ".join(f"w{i}" for i in range(400))  # two chunks per document
    list(service.batch_extract_entities([text, text], ["probe"], batch_size=3))
    assert [len(call) for call in service.model.calls] == [2, 2]


def test_document_chunks_never_split_across_model_calls(service):
    text = " ".join(f"w{i}" for i in range(1000))
    list(service.batch_extract_entities([text, "short doc"], ["probe"], batch_size=2))
    assert [len(call) for call in service.model.calls] == [3, 1]


def test_generator_is_lazy(service):
    texts = [f"doc{i} body" for i in range(5)]
    it = service.batch_extract_entities(texts, ["probe"], batch_size=2)
    next(it)
    assert len(service.model.calls) == 1
    next(it)
    assert len(service.model.calls) == 1
    next(it)
    assert len(service.model.calls) == 2
    assert len(list(it)) == 2
    assert len(service.model.calls) == 3


def test_empty_texts_yield_nothing(service):
    assert list(service.batch_extract_entities([], ["probe"])) == []
    assert service.model.calls == []


def test_global_span_invariant_through_stub(service):
    text = " ".join(f"w{i}" for i in range(1000))
    (result,) = service.batch_extract_entities([text], ["probe"])
    assert len(result.entities["probe"]) > 1
    for item in result.entities["probe"]:
        assert text[item.start : item.end] == item.text


def test_progress_reports_document_index(service):
    texts = ["alpha beta", "gamma delta", "epsilon zeta"]
    results = list(service.batch_extract_entities(texts, ["probe"]))
    assert [r.progress.current for r in results] == [1, 2, 3]
    assert [r.progress.length for r in results] == [3, 3, 3]
    assert [r.progress.progress for r in results] == [1 / 3, 2 / 3, 3 / 3]
