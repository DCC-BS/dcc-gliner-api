# ruff: noqa: S311, S603, C901 - local metering tool: fixed subprocess, non-crypto test data
"""Meter GPU memory of the entity scan, document by document.

Runs the real ``GlinerService`` in-process (no Ray, no HTTP) over a folder of
``.txt`` documents and/or a set of generated adversarial documents, and for
every model call records the batch it actually ran (shape the encoder saw), the
peak memory it reached, and what it was predicted to need. After every
document it records what is still allocated and reserved, so growth that
survives a document — a leak — shows up as drift across passes.

A smaller card can be emulated on a bigger one: ``--emulate-gib 11`` starts a
ballast process that occupies the rest of the card before the model loads, so
the service sizes its budget from the same free memory a real 11 GiB card
would report.

Usage:
    .venv/bin/python tools/vram_meter.py --docs DIR [--synthetic] \
        [--schema small|large|both] [--passes 3] [--emulate-gib 11] \
        [--out results.jsonl]
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import string
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

MIB = 1024**2
GIB = 1024**3

SMALL_SCHEMA = {
    "person": "Name einer natürlichen Person",
    "ort": "Name einer Gemeinde, Stadt oder eines Landes",
    "organisation": "Name einer Firma, Behörde oder Organisation",
}

#: The kind of schema the upload endpoint was built for: two dozen labels,
#: each with a paragraph describing it.
LARGE_SCHEMA = {
    label: (
        f"{text} Erfasse jede Erwähnung im Dokument, auch wenn sie abgekürzt, falsch geschrieben, "
        "in einer Tabelle, Fusszeile oder Aufzählung steht oder nur teilweise genannt wird. "
        "Nicht erfassen: generische Rollenbezeichnungen ohne Bezug zu einer bestimmten Instanz."
    )
    for label, text in {
        "vorname": "Vorname einer natürlichen Person.",
        "nachname": "Nachname oder Familienname einer natürlichen Person.",
        "geburtsdatum": "Geburtsdatum einer Person in beliebigem Datumsformat.",
        "strasse": "Strassenname mit oder ohne Hausnummer.",
        "postleitzahl": "Schweizer oder ausländische Postleitzahl.",
        "ort": "Wohnort, Gemeinde oder Stadt einer Person.",
        "telefonnummer": "Telefon- oder Mobilnummer in beliebiger Schreibweise.",
        "email": "E-Mail-Adresse einer Person oder Organisation.",
        "iban": "IBAN oder Kontonummer einer Bank.",
        "ahv_nummer": "AHV-Nummer oder Sozialversicherungsnummer (756.xxxx.xxxx.xx).",
        "pass_nummer": "Nummer eines Passes, einer Identitätskarte oder eines Ausländerausweises.",
        "autokennzeichen": "Kontrollschild eines Fahrzeugs, z. B. BS 123 456.",
        "aktenzeichen": "Geschäfts-, Verfahrens- oder Aktennummer einer Behörde oder eines Gerichts.",
        "arbeitgeber": "Name des Arbeitgebers einer Person.",
        "beruf": "Berufsbezeichnung oder Funktion einer Person.",
        "gesundheit": "Diagnose, Krankheit, Medikation oder Behandlung einer Person.",
        "religion": "Religionszugehörigkeit oder Konfession einer Person.",
        "nationalitaet": "Staatsangehörigkeit oder Herkunft einer Person.",
        "geschlecht": "Geschlecht einer Person, wenn ausdrücklich genannt.",
        "alter": "Alter einer Person in Jahren.",
        "ip_adresse": "IPv4- oder IPv6-Adresse.",
        "url": "Persönliche Webadresse oder Profil-URL.",
        "unterschrift": "Hinweis auf eine Unterschrift oder Visum einer Person.",
        "kind": "Name oder Hinweis auf ein minderjähriges Kind.",
    }.items()
}


# ── Synthetic documents ──────────────────────────────────────────────────────

PROSE = (
    "Der Regierungsrat des Kantons Basel-Stadt hat an seiner Sitzung vom 3. März das Geschäft beraten. "
    "Frau Andrea Müller-Schmid, wohnhaft an der Hauptstrasse 12 in 4058 Basel, hat am 12. Januar ein Gesuch "
    "eingereicht. Herr Dr. Peter Meier vom Bau- und Verkehrsdepartement nahm dazu Stellung. "
)


def _random_token(rng: random.Random, alphabet: str, length: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(length))


def _safelink(rng: random.Random) -> str:
    target = "https%3A%2F%2Fwww.bs.ch%2F" + _random_token(rng, string.ascii_lowercase, 40)
    data = _random_token(rng, string.ascii_letters + string.digits + "%", 180)
    sdata = _random_token(rng, string.ascii_letters + string.digits + "%", 60)
    return f"https://eur01.safelinks.protection.outlook.com/?url={target}&data={data}&sdata={sdata}&reserved=0"


def synthetic_documents(seed: int = 7) -> dict[str, str]:
    """Documents whose later chunks tokenise far longer than their first.

    The scan plan measures only the first chunk. Each of these opens with
    ordinary prose (cover page, letter head) and then turns into text whose
    "words" are long runs of subword tokens.
    """
    rng = random.Random(seed)
    head = PROSE * 12  # ~ 500 words of plain prose: the first chunk looks harmless

    b64 = string.ascii_letters + string.digits + "+/"
    base64_blob = " ".join(_random_token(rng, b64, 76) for _ in range(800))

    safelinks = " ".join(f"Gesendet von Andrea Müller. Link: {_safelink(rng)} Bitte prüfen." for _ in range(300))

    glued = " ".join("".join(word.capitalize() for word in (PROSE * 3).split())[:600] for _ in range(80))

    hexes = " ".join(_random_token(rng, "0123456789abcdef", 64) for _ in range(1500))

    numbers = "\n".join(
        f"{rng.randint(1000, 9999)}  {rng.randint(1, 9)}'{rng.randint(100, 999)}'{rng.randint(100, 999)}.{rng.randint(10, 99)}  "
        f"CH{rng.randint(10**18, 10**19 - 1)}  756.{rng.randint(1000, 9999)}.{rng.randint(1000, 9999)}.{rng.randint(10, 99)}"
        for _ in range(1500)
    )

    form_lines = "\n".join(f"Feld {i}: " + "_" * rng.randint(40, 120) for i in range(1500))

    return {
        "syn_prose_only": PROSE * 120,
        "syn_base64_after_cover": head + base64_blob,
        "syn_safelinks_after_cover": head + safelinks,
        "syn_glued_words_after_cover": head + glued,
        "syn_hex_hashes_after_cover": head + hexes,
        "syn_number_table": head + numbers,
        "syn_form_underscores": head + form_lines,
    }


# ── Ballast: emulate a smaller card ─────────────────────────────────────────

BALLAST_CODE = """
import sys, torch
target_free = int(sys.argv[1])
torch.cuda.init()
free, _total = torch.cuda.mem_get_info()
held = []
step = 256 * 1024**2
while True:
    free, _ = torch.cuda.mem_get_info()
    if free - step < target_free:
        break
    held.append(torch.empty(step, dtype=torch.uint8, device='cuda'))
free, _ = torch.cuda.mem_get_info()
print(f'ready {free}', flush=True)
sys.stdin.read()
"""


def start_ballast(emulate_gib: float) -> subprocess.Popen[str]:
    """Occupy the card until only ``emulate_gib`` is free for this process."""
    proc = subprocess.Popen(
        [sys.executable, "-c", BALLAST_CODE, str(int(emulate_gib * GIB))],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline() if proc.stdout else ""
    if not line.startswith("ready"):
        raise RuntimeError(f"ballast failed: {line!r}")
    print(f"ballast up: {int(line.split()[1]) / GIB:.2f} GiB left free on the card", flush=True)
    return proc


# ── Metering ─────────────────────────────────────────────────────────────────


class Meter:
    """Wraps a live service and records what every model call cost."""

    def __init__(self, service: Any) -> None:
        import torch

        self.torch = torch
        self.service = service
        self.calls: list[dict[str, Any]] = []
        self._shapes: list[tuple[int, int]] = []

        service.model.encoder.register_forward_pre_hook(self._record_shape, with_kwargs=True)

        original_scan = service._scan_chunks

        def metered_scan(chunk_texts, entity_types, *, batch_size, threshold):
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            self._shapes = []
            started = time.perf_counter()
            error = None
            try:
                return original_scan(chunk_texts, entity_types, batch_size=batch_size, threshold=threshold)
            except torch.OutOfMemoryError as exc:
                error = f"OOM: {str(exc).splitlines()[0]}"
                raise
            finally:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated()
                predicted = [service._cost.memory_for(batch, seq) for batch, seq in self._shapes]
                self.calls.append({
                    "chunks": len(chunk_texts),
                    "labels": len(entity_types),
                    "planned_batch": batch_size,
                    "encoder_shapes": list(self._shapes),
                    "max_seq": max((s for _, s in self._shapes), default=0),
                    "peak_activation_mib": (peak - before) / MIB,
                    "predicted_for_actual_shape_mib": max(predicted, default=0) / MIB,
                    "seconds": time.perf_counter() - started,
                    "error": error,
                })

        service._scan_chunks = metered_scan

    def _record_shape(self, _module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        if input_ids is not None:
            self._shapes.append(tuple(input_ids.shape))

    def snapshot(self) -> dict[str, float]:
        torch = self.torch
        torch.cuda.synchronize()
        free, _total = torch.cuda.mem_get_info()
        return {
            "allocated_mib": torch.cuda.memory_allocated() / MIB,
            "reserved_mib": torch.cuda.memory_reserved() / MIB,
            "device_free_mib": free / MIB,
            "live_cuda_tensors": sum(1 for obj in gc.get_objects() if isinstance(obj, torch.Tensor) and obj.is_cuda),
        }


def chunk_sequence_lengths(service: Any, text: str, entity_types: Any) -> list[int]:
    """Tokens every chunk of ``text`` is scanned in, schema included.

    Measured with the model's own collator, independently of the estimate the
    service plans with, so the two can be compared.
    """
    return [service._tokenized_length(entity_types, chunk.text) for chunk in service._chunks_of(text)]


def load_documents(folder: Path | None, synthetic: bool, limit_words: int | None) -> dict[str, str]:
    documents: dict[str, str] = {}
    if folder:
        for path in sorted(folder.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="replace")
            if limit_words:
                text = " ".join(text.split(" ")[:limit_words])
            documents[path.stem] = text
    if synthetic:
        documents.update(synthetic_documents())
    return documents


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, help="folder of .txt documents")
    parser.add_argument("--synthetic", action="store_true", help="add generated adversarial documents")
    parser.add_argument("--schema", choices=["small", "large", "both"], default="both")
    parser.add_argument("--passes", type=int, default=2, help="times to run the whole corpus (leak check)")
    parser.add_argument("--emulate-gib", type=float, help="leave only this much free on the card")
    parser.add_argument(
        "--path",
        choices=["single", "batch"],
        default="single",
        help="single: extract_entities per doc; batch: batch_extract_entities over the corpus",
    )
    parser.add_argument("--limit-words", type=int)
    parser.add_argument("--only", help="comma-separated document names to run")
    parser.add_argument("--out", type=Path, default=Path("vram_meter.jsonl"))
    args = parser.parse_args()

    ballast = start_ballast(args.emulate_gib) if args.emulate_gib else None

    import torch

    from dcc_gliner_api.services.gliner_service import DEFAULT_BATCH_SIZE, GlinerService
    from dcc_gliner_api.services.memory_plan import plan_scan

    service = GlinerService()
    meter = Meter(service)
    baseline = meter.snapshot()
    print(
        f"cost {service._cost.bytes_per_token_squared} B/token² ({service._cost.source}), "
        f"budget {service._budget_bytes / GIB:.2f} GiB, baseline {baseline}",
        flush=True,
    )

    documents = load_documents(args.docs, args.synthetic, args.limit_words)
    if args.only:
        wanted = set(args.only.split(","))
        documents = {k: v for k, v in documents.items() if k in wanted}
    schemas = {"small": SMALL_SCHEMA, "large": LARGE_SCHEMA}
    schema_names = ["small", "large"] if args.schema == "both" else [args.schema]

    with args.out.open("w", encoding="utf-8") as out:
        out.write(
            json.dumps({
                "kind": "baseline",
                **baseline,
                "cost_bytes_per_token_sq": service._cost.bytes_per_token_squared,
                "cost_source": service._cost.source,
                "budget_gib": service._budget_bytes / GIB,
            })
            + "\n"
        )
        for pass_index in range(args.passes):
            for schema_name in schema_names:
                entity_types = schemas[schema_name]
                items = list(documents.items())
                if args.path == "batch":
                    items = [("__batch__", None)]
                for name, text in items:
                    meter.calls.clear()
                    started = time.perf_counter()
                    error = None
                    seq_lengths: list[int] = []
                    try:
                        if text is None:
                            list(service.batch_extract_entities(list(documents.values()), entity_types))
                        else:
                            seq_lengths = chunk_sequence_lengths(service, text, entity_types)
                            service.extract_entities(text, entity_types)
                    except torch.OutOfMemoryError as exc:
                        error = f"OOM: {str(exc).splitlines()[0]}"
                    except Exception as exc:  # metering: record and keep going
                        error = f"{type(exc).__name__}: {exc}"
                        traceback.print_exc()
                    gc.collect()
                    after = meter.snapshot()

                    planned = None
                    if seq_lengths:
                        # The plan the longest chunk gets; slices of shorter chunks may be larger.
                        plan = plan_scan(
                            service._budget_bytes,
                            max(seq_lengths),
                            DEFAULT_BATCH_SIZE,
                            len(entity_types),
                            service._cost,
                        )
                        planned = {"batch": plan.batch_size, "groups": plan.schema_groups}
                    record = {
                        "kind": "document",
                        "pass": pass_index,
                        "schema": schema_name,
                        "doc": name,
                        "words": len(text.split()) if text else None,
                        "chunks": len(seq_lengths) or None,
                        "first_chunk_seq": seq_lengths[0] if seq_lengths else None,
                        "max_chunk_seq": max(seq_lengths) if seq_lengths else None,
                        "plan": planned,
                        "calls": len(meter.calls),
                        "max_encoder_seq": max((c["max_seq"] for c in meter.calls), default=0),
                        "peak_activation_mib": max((c["peak_activation_mib"] for c in meter.calls), default=0),
                        "predicted_mib_at_actual_shape": max(
                            (c["predicted_for_actual_shape_mib"] for c in meter.calls), default=0
                        ),
                        "seconds": time.perf_counter() - started,
                        "error": error,
                        **{f"after_{k}": v for k, v in after.items()},
                        "allocated_drift_mib": after["allocated_mib"] - baseline["allocated_mib"],
                        "reserved_drift_mib": after["reserved_mib"] - baseline["reserved_mib"],
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(
                        f"[p{pass_index} {schema_name:5}] {name:30} chunks={record['chunks']} "
                        f"seq first/max={record['first_chunk_seq']}/{record['max_chunk_seq']} "
                        f"plan={planned} peak={record['peak_activation_mib']:.0f}MiB "
                        f"alloc_drift={record['allocated_drift_mib']:+.1f} "
                        f"reserved={after['reserved_mib']:.0f} free={after['device_free_mib']:.0f} "
                        f"tensors={after['live_cuda_tensors']} {error or ''}",
                        flush=True,
                    )

    if ballast:
        ballast.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
