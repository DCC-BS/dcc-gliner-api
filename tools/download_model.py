"""Pre-download the GLiNER2 model into the image so the first request needs no network.

Mirrors DCC-BS/bentoml-faster-whisper's build-stage model baking. Runs in the
build stage and writes into $HF_HOME so the runtime stage can COPY it over.
"""
import os

from huggingface_hub import snapshot_download

REPO = os.environ.get("GLINER_MODEL", "fastino/gliner2-base-v1")

snapshot_download(repo_id=REPO)
