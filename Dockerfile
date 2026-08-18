# Stage 1: Build
FROM python:3.13-slim AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python

WORKDIR /app

# Dependency caching layer: only invalidated when the lockfile changes
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Pre-download the GLiNER2 model into the image so the first request needs no
# network. Placed before `COPY . /app` and bind-mounting only the script, so
# this heavy layer is cached independently of application source changes
# (re-runs only when the script or the deps change).
ARG GLINER_MODEL=fastino/gliner2-multi-v1
RUN --mount=type=bind,source=tools/download_model.py,target=/tmp/download_model.py \
    if [ -n "${GLINER_MODEL}" ]; then \
      HF_HOME=/opt/models GLINER_MODEL="${GLINER_MODEL}" /app/.venv/bin/python /tmp/download_model.py; \
    fi

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Stage 2: Runtime
FROM python:3.13-slim

ENV TZ=Europe/Zurich
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# venv is built against the system Python from python:3.13-slim, so the runtime
# stage gets it via COPY of /app.
COPY --from=build /app /app

# Baked model cache. Copied to the runtime HF cache location so the model is
# found with no env var. When compose mounts the (empty) hf-cache named volume
# over this path, Docker seeds the volume from these contents on first start.
COPY --from=build /opt/models /app/.cache/huggingface

ENV PATH="/app/.venv/bin:$PATH"

# Point the HuggingFace cache at the baked model location so no network is needed
ENV HF_HOME=/app/.cache/huggingface

# Ray Serve's HTTP proxy must listen on all interfaces for Docker port mapping
ENV RAY_SERVE_DEFAULT_HTTP_HOST=0.0.0.0

EXPOSE 8000

CMD ["ray", "serve", "run", "src.dcc_gliner_api.app:app"]
