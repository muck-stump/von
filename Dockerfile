FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl gcc g++ python3-dev && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /app
WORKDIR /app

# uv.toml configures index-strategy = "unsafe-best-match" and the IBM ppc64le wheel index;
# delete uv.lock so uv re-resolves cleanly for this architecture
RUN rm -f uv.lock && uv sync --no-dev

# Pre-download model weights from HF Hub into the local checkpoint directory so
# the image is self-contained and needs no network access at inference time.
RUN mkdir -p checkpoints/von-1.2 && \
    .venv/bin/python -c " \
from huggingface_hub import hf_hub_download, snapshot_download; \
snapshot_download(repo_id='wfzyx/von', local_dir='checkpoints/von-1.2', ignore_patterns=['*.md','*.txt']); \
print('Model weights baked in.')"

# Run as non-root (OpenShift arbitrary-UID compatible: group 0 is always granted)
RUN chown -R 1001:0 /app && chmod -R g=u /app
USER 1001

EXPOSE 8000

CMD ["/app/.venv/bin/von", "serve", "--host", "0.0.0.0", "--port", "8000"]
