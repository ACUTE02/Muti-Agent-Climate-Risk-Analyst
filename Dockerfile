# syntax=docker/dockerfile:1.7
#
# Local image for the Climate Risk Analyst API.
#
# Build:
#   docker build --secret id=gemini_key,env=GEMINI_API_KEY -t climate-risk-analyst .
# Run:
#   docker run --rm -p 8000:8000 -e GEMINI_API_KEY=your-key climate-risk-analyst
#
# Two rules this file exists to hold, and the tension between them:
#
#   1. The ChromaDB index is baked in at BUILD time, per the project's standing
#      architecture decision — no zero-cost host offers persistent disk, so the
#      index cannot be built on first boot onto a mounted volume.
#   2. The API key is supplied at RUN time and is never baked into the image.
#
# Building the index needs a key, so those two rules collide. They are resolved
# with a BuildKit secret mount rather than an ARG: an ARG would persist the key
# in image history, which is exactly what rule 2 forbids. The secret is visible
# only to the RUN step that mounts it and leaves nothing behind in any layer.
#
# No deployment-platform configuration lives here — no Hugging Face Spaces
# frontmatter, no Cloud Run YAML. Where this gets deployed is a later decision.

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# curl is used by HEALTHCHECK; build-essential is needed by some wheels that
# have no manylinux build for slim, and is dropped again in the same layer.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so a code change does not re-resolve the whole tree.
# Note this is requirements.txt only, never requirements-training.txt:
# tensorflow and matplotlib belong to the LSTM path, which the API never imports.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential

COPY . .

# Regenerate every derived artifact inside the image: weather data, ENSO series,
# the runtime scaler and SPI-3 parameters, the per-horizon Ridge models, and the
# ChromaDB index. This is the same `scripts.setup` a developer runs after
# cloning — the image and a local checkout are set up by identical code, so they
# cannot drift apart.
#
# The build reaches the network for Open-Meteo, NOAA and the corpus documents.
# If the secret is absent the index step is skipped with a message and the build
# still succeeds; the resulting image serves /health as "degraded" and answers
# everything except retrieval-backed questions.
RUN --mount=type=secret,id=gemini_key,required=false \
    sh -c 'if [ -f /run/secrets/gemini_key ]; then \
             export GEMINI_API_KEY="$(cat /run/secrets/gemini_key)"; \
           fi; \
           python -m scripts.setup'

# Fail the build rather than ship an image whose forecast artifacts are missing.
# The ChromaDB index is deliberately not required here, since a keyless build is
# a supported outcome above.
RUN python -c "import sys; \
from scripts.setup import verify; \
ok, missing = verify(); \
blocking = [m for m in missing if 'chroma' not in m and 'chunks' not in m]; \
print('missing after setup:', missing or 'nothing'); \
print('BLOCKING:', blocking) or sys.exit(1) if blocking else print('forecast artifacts present')"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# No --reload: that is a development convenience and would watch the whole tree.
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
