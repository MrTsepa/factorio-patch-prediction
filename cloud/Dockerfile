# Container for running the POC training on a cloud GPU.
#
# Build for linux/amd64 (GPU hosts). On Linux x86_64, torch's default PyPI
# wheels are CUDA-enabled, so the container picks up the GPU automatically when
# run with the NVIDIA runtime (`--gpus all` locally, or any cloud GPU host).
#
#   docker buildx build --platform linux/amd64 -f cloud/Dockerfile -t factorio-patch:latest --load .
#
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl bash && rm -rf /var/lib/apt/lists/*

# uv (matches the local toolchain; see memory/use-uv-run)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Dependency layer (cached unless pyproject/lock change).
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

# App files.
COPY data/seed_urls.txt ./data/seed_urls.txt
COPY scripts ./scripts
COPY cloud/entrypoint.sh ./cloud/entrypoint.sh
RUN chmod +x cloud/entrypoint.sh

# Tunable defaults (override with `-e EPOCHS=...` etc.).
ENV EPOCHS=24 \
    BATCH_SIZE=48 \
    EMPTY_WEIGHT=0.12 \
    MAX_VOCAB=160 \
    DATA_DIR=/workspace/data \
    OUT_DIR=/workspace/runs/poc_cloud \
    OUTPUTS_DIR=/workspace/outputs

# Mount a volume at /workspace to keep checkpoints + predictions after the run.
VOLUME ["/workspace"]
ENTRYPOINT ["bash", "cloud/entrypoint.sh"]
