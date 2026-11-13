# ruff: noqa: S310, S603, S607, C901 - local load tool: fixed commands and caller-given URL
"""Drive a running server with documents and sample its GPU memory over time.

Submits every document through ``/extract_entities/async/upload`` (the path
the production caller uses), polls until done, collects the result, and
repeats for ``--passes``. A sampler thread records the GPU memory every
process on the card uses (``nvidia-smi``) once per second, so a replica whose
footprint keeps growing across passes shows up as a rising line rather than a
plateau.

Usage:
    .venv/bin/python tools/serve_load.py --url http://localhost:8000 \
        --docs DIR [--synthetic] [--schema small|large] [--passes 3] \
        [--gpu 0] [--out serve_load.jsonl]
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from vram_meter import LARGE_SCHEMA, SMALL_SCHEMA, load_documents


def gpu_used_mib(gpu: int) -> dict[str, int]:
    """Memory each compute process uses on ``gpu``, by pid."""
    uuid_out = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    used: dict[str, int] = {}
    for line in out.strip().splitlines():
        gpu_uuid, pid, mib = (part.strip() for part in line.split(","))
        if gpu_uuid == uuid_out:
            used[pid] = int(mib)
    return used


def post_upload(url: str, body: dict) -> str:
    boundary = uuid.uuid4().hex
    payload = gzip.compress(json.dumps(body, ensure_ascii=False).encode())
    data = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="doc.json.gz"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{url}/extract_entities/async/upload",
        data=data,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["task_id"]


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--docs", type=Path)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--only", help="comma-separated document names to run")
    parser.add_argument("--schema", choices=["small", "large"], default="large")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("serve_load.jsonl"))
    args = parser.parse_args()

    documents = load_documents(args.docs, args.synthetic, None)
    if args.only:
        wanted = set(args.only.split(","))
        documents = {k: v for k, v in documents.items() if k in wanted}
    entity_types = SMALL_SCHEMA if args.schema == "small" else LARGE_SCHEMA

    out = args.out.open("w", encoding="utf-8")
    lock = threading.Lock()
    stop = threading.Event()
    started = time.monotonic()

    def write(record: dict) -> None:
        with lock:
            out.write(json.dumps({"t": round(time.monotonic() - started, 2), **record}) + "\n")
            out.flush()

    def sample() -> None:
        while not stop.is_set():
            try:
                write({"kind": "gpu", "used_mib": gpu_used_mib(args.gpu)})
            except Exception as exc:  # sampling must not stop the run
                write({"kind": "gpu_error", "error": str(exc)})
            stop.wait(1.0)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    for pass_index in range(args.passes):
        for name, text in documents.items():
            t0 = time.monotonic()
            task_id = post_upload(args.url, {"text": text, "entity_types": entity_types, "threshold": 0.5})
            while True:
                state = get_json(f"{args.url}/task/{task_id}")
                if state["status"] in ("finished", "failed"):
                    break
                time.sleep(0.5)
            mentions = None
            if state["status"] == "finished":
                result = get_json(f"{args.url}/resource/{state['resource_id']}")
                mentions = sum(len(v) for v in result["entities"].values())
            used = gpu_used_mib(args.gpu)
            write({
                "kind": "doc",
                "pass": pass_index,
                "doc": name,
                "status": state["status"],
                "error": state.get("error"),
                "mentions": mentions,
                "seconds": round(time.monotonic() - t0, 2),
                "used_mib": used,
            })
            print(
                f"[p{pass_index}] {name:30} {state['status']:8} {time.monotonic() - t0:6.1f}s "
                f"gpu={used} {(state.get('error') or '')[:90]}",
                flush=True,
            )

    stop.set()
    sampler.join()
    out.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
