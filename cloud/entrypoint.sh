#!/usr/bin/env bash
# Runs the full POC end-to-end inside the container: (prepare data if needed) ->
# train -> eval. Writes checkpoints + prediction PNGs under /workspace so they
# survive after the container exits. All knobs come from env vars (see Dockerfile).
set -euo pipefail
cd /app

DATA_DIR="${DATA_DIR:-/workspace/data}"
OUT_DIR="${OUT_DIR:-/workspace/runs/poc_cloud}"
OUTPUTS_DIR="${OUTPUTS_DIR:-/workspace/outputs}"
EPOCHS="${EPOCHS:-24}"
BATCH_SIZE="${BATCH_SIZE:-48}"
EMPTY_WEIGHT="${EMPTY_WEIGHT:-0.12}"
MAX_VOCAB="${MAX_VOCAB:-160}"
SLEEP_MIN="${SLEEP_MIN:-2}"
SLEEP_MAX="${SLEEP_MAX:-5}"

mkdir -p "$DATA_DIR/processed" "$OUT_DIR" "$OUTPUTS_DIR"

echo "== environment =="
uv run python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'))"

DATASET="$DATA_DIR/processed/dataset.pt"
if [ ! -f "$DATASET" ]; then
  echo "== prepare data (no dataset.pt found) =="
  uv run python -m factorio_patches.download_factoriobin \
    --urls data/seed_urls.txt --out "$DATA_DIR/raw/factoriobin" \
    --sleep-min "$SLEEP_MIN" --sleep-max "$SLEEP_MAX"
  uv run python -m factorio_patches.blueprint_decode \
    --raw "$DATA_DIR/raw/factoriobin" --out "$DATA_DIR/decoded"
  uv run python -m factorio_patches.blueprint_extract \
    --decoded "$DATA_DIR/decoded" --out "$DATA_DIR/processed/blueprints.jsonl"
  uv run python -m factorio_patches.vocab \
    --blueprints "$DATA_DIR/processed/blueprints.jsonl" \
    --out "$DATA_DIR/processed/vocab.json" --max-vocab "$MAX_VOCAB"
  uv run python -m factorio_patches.dataset \
    --blueprints "$DATA_DIR/processed/blueprints.jsonl" \
    --vocab "$DATA_DIR/processed/vocab.json" --out "$DATASET" \
    --max-dim 384 --min-dim 24 --min-entities 8
else
  echo "== using existing dataset: $DATASET =="
fi

echo "== train =="
uv run python -m factorio_patches.train \
  --data "$DATASET" --out "$OUT_DIR" \
  --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --empty-weight "$EMPTY_WEIGHT" \
  --device auto

echo "== eval (test split) =="
uv run python -m factorio_patches.eval \
  --checkpoint "$OUT_DIR/best.pt" --data "$DATASET" \
  --out "$OUTPUTS_DIR/demo_predictions" --num 20 --split test --device auto

echo "DONE. checkpoints -> $OUT_DIR ; predictions -> $OUTPUTS_DIR/demo_predictions"
