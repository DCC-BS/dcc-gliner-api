"""Probe the encoder context window of GLiNER2.extract_entities.

Calls the raw model (no HTTP/Ray layer) with increasingly long texts and
reports where extraction quality breaks down. The signal is whether a
probe entity ("Max Mustermann") placed at the very end of the text is
still found, since the encoder budget is shared by schema and text tokens
(deberta-v3-base, max_position_embeddings=512) and quality degrades
rather than erroring past it.

Phase 1 sweeps a coarse ladder of target subword-token counts, phase 2
binary-searches between the last working and first broken length.

Usage:
    uv run python tools/probe_context_window.py [--model ID] [--device cuda|cpu]
        [--precision 16] [--ladder 128,256,384,512,768,1024,1536,2048]
"""

import argparse
import os
import time

import torch
from gliner2.inference.engine import GLiNER2

PROBE_NAME = "Max Mustermann"
PROBE_SENTENCE = f"Am Ende des Schreibens meldete sich {PROBE_NAME} persönlich."
FILLER = [
    "Das Dokument enthält mehrere Abschnitte mit allgemeinen Hinweisen.",
    "Die Verwaltung prüft die Unterlagen und ergänzt fehlende Angaben.",
    "Alle Beteiligten erhalten eine schriftliche Bestätigung der Änderungen.",
    "Die Frist für die Rückmeldung wurde erneut verschoben.",
    "Der Bericht beschreibt den Ablauf der Bearbeitung in groben Zügen.",
    "Zusätzliche Informationen können bei der Stelle angefordert werden.",
]
DEFAULT_LADDER = [128, 256, 384, 512, 768, 1024, 1536, 2048]
PROBE_EVERY = 6

_word_cache: dict[str, int] = {}


def _count_words(word: str, tokenizer) -> int:
    if word not in _word_cache:
        _word_cache[word] = len(tokenizer.tokenize(word))
    return _word_cache[word]


def count_tokens(text: str, tokenizer, splitter) -> int:
    return sum(_count_words(w, tokenizer) for w, _, _ in splitter(text, lower=True))


def build_text(target: int, tokenizer, splitter) -> tuple[str, int, int]:
    """Assemble filler + probe sentences totalling <= target subword tokens,
    ending on a probe sentence. Returns (text, subword tokens, word tokens)."""
    probe_len = count_tokens(PROBE_SENTENCE, tokenizer, splitter)
    sentences: list[str] = []
    total = 0
    i = 0
    while True:
        sent = PROBE_SENTENCE if i % (PROBE_EVERY + 1) == PROBE_EVERY else FILLER[i % len(FILLER)]
        c = count_tokens(sent, tokenizer, splitter)
        if total + c + probe_len > target:
            break
        sentences.append(sent)
        total += c
        i += 1
    sentences.append(PROBE_SENTENCE)
    total += probe_len
    text = " ".join(sentences)
    return text, total, sum(1 for _ in splitter(text, lower=True))


def probe_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    start = text.find(PROBE_NAME)
    while start != -1:
        spans.append((start, start + len(PROBE_NAME)))
        start = text.find(PROBE_NAME, start + 1)
    return spans


def check(model, tokenizer, splitter, target: int) -> dict:
    text, tokens, words = build_text(target, tokenizer, splitter)
    spans = probe_spans(text)
    t0 = time.perf_counter()
    status = "OK"
    result: dict = {}
    try:
        result = model.extract_entities(
            text,
            ["person"],
            include_confidence=True,
            include_spans=True,
            max_len=2048,
            format_results=False,
        )
    except Exception as exc:
        status = f"RAISED ({type(exc).__name__})"
    seconds = time.perf_counter() - t0

    found = 0
    tail_found = False
    if status == "OK":
        if result == {}:
            status = "SILENT-FAIL"
        else:
            entities = []
            if isinstance(result, dict):
                for group in result.get("entities", []):
                    entities.extend(group.get("person", []))
            for m in entities:
                for ps, pe in spans:
                    if m["start"] < pe and m["end"] > ps:
                        found += 1
                        break
            tail_start = text.rfind(PROBE_NAME)
            tail_end = tail_start + len(PROBE_NAME)
            tail_found = any(m["start"] < tail_end and m["end"] > tail_start for m in entities)
    return {
        "target": target,
        "tokens": tokens,
        "words": words,
        "chars": len(text),
        "probes": f"{found}/{len(spans)}",
        "status": status,
        "tail": tail_found,
        "seconds": seconds,
    }


def run(model, tokenizer, splitter, ladder: list[int], precision: int) -> list[dict]:
    rows = [check(model, tokenizer, splitter, t) for t in ladder]

    working = [r["target"] for r in rows if r["tail"]]
    if not working:
        return rows
    if len(working) == len(rows):
        return rows

    lo = max(working)
    hi = min(r["target"] for r in rows if r["target"] > lo)
    while hi - lo > precision:
        mid = (lo + hi) // 2
        row = check(model, tokenizer, splitter, mid)
        rows.append(row)
        if row["tail"]:
            lo = mid
        else:
            hi = mid
    rows.sort(key=lambda r: r["target"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("GLINER_MODEL", "fastino/gliner2-base-v1"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", type=int, default=16)
    parser.add_argument("--ladder", default=",".join(map(str, DEFAULT_LADDER)))
    args = parser.parse_args()
    ladder = [int(t) for t in args.ladder.split(",")]

    model = GLiNER2.from_pretrained(args.model).to(args.device)
    print(f"model: {args.model} | device: {args.device}")
    tokenizer = model.processor.tokenizer
    splitter = model.processor.word_splitter

    rows = run(model, tokenizer, splitter, ladder, args.precision)

    print(f"\n{'target':>7} {'tokens':>7} {'words':>6} {'chars':>7} {'probes':>7} {'tail':>5} {'secs':>6}  status")
    for r in rows:
        print(
            f"{r['target']:>7} {r['tokens']:>7} {r['words']:>6} {r['chars']:>7} "
            f"{r['probes']:>7} {'yes' if r['tail'] else 'NO':>5} {r['seconds']:>6.2f}  {r['status']}"
        )

    working = [r for r in rows if r["tail"]]
    if working:
        best = max(working, key=lambda r: r["tokens"])
        print(
            f"\nmax working length: {best['tokens']} text subword tokens ({best['words']} words, {best['chars']} chars)"
        )
        broken = [r for r in rows if not r["tail"] and r["target"] > best["target"]]
        if broken:
            first = min(broken, key=lambda r: r["tokens"])
            print(f"first broken length: {first['tokens']} tokens ({first['status']})")
        else:
            print("no failure within the swept range")
    else:
        print("\nno length worked; extraction never found the tail probe")


if __name__ == "__main__":
    main()
