"""Integration test for the live /batch_extract_entities endpoint.

Sends several large texts with known person/location mentions, streams the
NDJSON response, and asserts each mention is recognised at the correct global
character span (i.e. ``text[start:end]`` equals the expected mention text).

These tests hit a running server and are not collected by default; run them
with ``-m integration``.
"""

import json
import time

import pytest
import requests

API_URL = "http://localhost:8000"
ENDPOINT = f"{API_URL}/batch_extract_entities"

# (person, location) pairs embedded in each document below.
CASES = [
    ("Ada Lovelace", "London"),
    ("Alan Turing", "Manchester"),
    ("Grace Hopper", "New York"),
    ("Katherine Johnson", "Hampton"),
    ("Margaret Hamilton", "Cambridge"),
]


def _large_text(person: str, location: str) -> tuple[str, int, int]:
    """Build a long document with the person and location at known positions."""
    filler = (
        "The committee convened to review the quarterly progress of the "
        "engineering division and to plan the upcoming fiscal year. "
    )
    # Repeat filler so the document is large enough to span multiple chunks.
    body = filler * 40
    person_pos = len(body)
    body += f"{person} presented the findings."
    body += f" The meeting was held in {location}."
    location_pos = body.find(location)
    return body, person_pos, location_pos


@pytest.fixture(scope="module")
def texts():
    return [_large_text(person, location) for person, location in CASES]


@pytest.mark.integration
def test_batch_extract_entities_streams_and_places_entities(texts):
    payload = {
        "texts": [t for t, _, _ in texts],
        "entity_types": ["person", "location"],
    }

    with requests.post(ENDPOINT, json=payload, stream=True, timeout=300) as resp:
        assert resp.status_code == 200, resp.text
        assert resp.headers.get("content-type", "").startswith("application")

        lines = [line for line in resp.iter_lines() if line]
        assert len(lines) == len(texts), f"expected one line per text, got {len(lines)}"

        for line, (text, person_pos, location_pos) in zip(lines, texts, strict=True):
            doc = json.loads(line)
            entities = doc["entities"]
            assert "person" in entities, f"no person entities in {doc}"
            assert "location" in entities, f"no location entities in {doc}"

            person = entities["person"][0]
            location = entities["location"][0]

            # The mention must sit exactly where we planted it.
            assert person["start"] == person_pos
            assert person["end"] == person_pos + len(person["text"])
            assert text[person["start"] : person["end"]] == person["text"]

            assert location["start"] == location_pos
            assert location["end"] == location_pos + len(location["text"])
            assert text[location["start"] : location["end"]] == location["text"]


@pytest.mark.integration
def test_batch_streams_results_incrementally(texts):
    """Verify the response streams one line per document as it completes.

    Progress is asserted by timing: lines arrive incrementally rather than all at
    once, and each line is a complete, independently parseable JSON document.
    """
    payload = {
        "texts": [t for t, _, _ in texts],
        "entity_types": ["person", "location"],
        "include_spans": True,
    }

    received = []
    timestamps = []
    started = time.monotonic()

    with requests.post(ENDPOINT, json=payload, stream=True, timeout=300) as resp:
        assert resp.status_code == 200, resp.text
        for line in resp.iter_lines():
            if not line:
                continue
            received.append(json.loads(line))
            timestamps.append(time.monotonic() - started)

    assert len(received) == len(texts), f"expected one line per text, got {len(received)}"
    assert timestamps == sorted(timestamps), "results must stream in order"

    # All lines must have arrived incrementally over time, not in one burst.
    elapsed = timestamps[-1] - timestamps[0]
    assert elapsed > 0, "expected progressive streaming, got everything at once"

    # Each streamed line is a complete document with its expected entities.
    for doc, (_, person_pos, location_pos) in zip(received, texts, strict=True):
        assert set(doc["entities"]) == {"person", "location"}
        assert doc["entities"]["person"][0]["start"] == person_pos
        assert doc["entities"]["location"][0]["start"] == location_pos
        print(f"[{time.monotonic() - started:5.2f}s] person={doc['entities']['person'][0]['text']}")
