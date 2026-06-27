# factorio-patch-inpaint (POC)

Proof-of-concept for **Factorio blueprint patch inpainting**: take a blueprint,
mask out a rectangular region, and predict the entities inside the hole — like
image inpainting, but over a 2D grid of Factorio entity anchors.

```
Given a 64×64 crop of a blueprint with a 16×16 rectangle replaced by MASK,
predict the entity type + direction in each masked cell.
```

This repo proves the **whole pipeline end-to-end on real data**: it discovers and
politely downloads FactorioBin blueprint books, decodes/extracts them, rasterizes
them to token grids, builds masked-patch supervised examples, trains a small
U-Net, and reports train/val/test metrics against trivial baselines with
before/target/prediction visualizations.

**→ See [DEMO.md](DEMO.md) for a visual demo of the trained model in action.**

---

## 1. What this project does

- Downloads Factorio blueprint **books** from a list of known FactorioBin URLs
  (politely: clear User-Agent, 2–5 s sleeps, SHA256 dedupe, on-disk cache).
- Decodes blueprint strings (`version-byte → base64 → zlib → JSON`, plus a raw-JSON
  fallback).
- Recursively extracts individual blueprints from books (skipping planners, empty
  and oversized blueprints).
- Rasterizes each blueprint into an integer **token grid** (`entity_name|direction`),
  normalized and rounded to integer cells.
- Builds a **masked-patch dataset** with a blueprint-level train/val/test split.
- Trains a small **embedding + U-Net** to reconstruct the masked patch, with the
  loss computed only inside the mask.
- Reports careful metrics (non-empty F1, entity-token accuracy, top-5) vs.
  always-EMPTY and always-majority-entity baselines.
- Renders `input / target / prediction / diff` PNGs.

## 2. What it does NOT do yet

- It does **not** run Factorio. Predictions are not validated in-engine.
- v0 models **entity anchors only** — one cell per entity, no multi-tile
  collision boxes.
- It ignores fluids, circuit wires, trains/schedules, recipes, modules,
  inventories, and construction order. Direction is a token suffix, not modeled
  geometrically.
- It does not guarantee valid blueprints (belt connectivity, underground pairing,
  etc.). The blueprint export (`--export-blueprint`) is experimental.

## 3. Install

Uses [`uv`](https://docs.astral.sh/uv/). From the repo root:

```bash
uv sync                 # creates .venv, installs torch + deps (Python 3.12)
uv run pytest           # optional: run the unit tests
```

All commands below are run through `uv run`.

## 4. Download a tiny dataset

`data/seed_urls.txt` already contains ~10 verified Factorio 1.1 books (belt
balancer + Nilaus base/megabase books). Add your own FactorioBin post URLs (5–50)
as needed — one per line.

```bash
uv run python -m factorio_patches.download_factoriobin \
  --urls data/seed_urls.txt --out data/raw/factoriobin \
  --sleep-min 2 --sleep-max 5
```

Re-running is safe: content is keyed by the SHA256 of the blueprint string and
cached, so nothing is re-downloaded.

## 5. Decode / process data

```bash
uv run python -m factorio_patches.blueprint_decode \
  --raw data/raw/factoriobin --out data/decoded

uv run python -m factorio_patches.blueprint_extract \
  --decoded data/decoded --out data/processed/blueprints.jsonl

uv run python -m factorio_patches.vocab \
  --blueprints data/processed/blueprints.jsonl \
  --out data/processed/vocab.json --max-vocab 160

uv run python -m factorio_patches.dataset \
  --blueprints data/processed/blueprints.jsonl \
  --vocab data/processed/vocab.json --out data/processed/dataset.pt \
  --crop-size 64 --mask-size 16 --max-dim 384 --min-dim 24 --min-entities 8
```

`scripts/make_tiny_dataset.py` runs decode→extract→vocab→dataset in one shot.

Inspect the data before training:

```bash
uv run python -m factorio_patches.rasterize \
  --blueprints data/processed/blueprints.jsonl --vocab data/processed/vocab.json --max-dim 384
uv run python -m factorio_patches.render \
  --dataset data/processed/dataset.pt --out outputs/samples --num 20 --split val
```

## 6. Train the POC model

```bash
uv run python -m factorio_patches.train \
  --data data/processed/dataset.pt --out runs/poc_001 \
  --epochs 20 --batch-size 48 --empty-weight 0.2
```

Each epoch prints val metrics + baselines, saves `last.pt`/`best.pt`, and writes
prediction PNGs to `runs/poc_001/preds/epoch_XX/`. The test set is evaluated with
the best checkpoint at the end.

## 7. Render predictions

```bash
uv run python -m factorio_patches.eval \
  --checkpoint runs/poc_001/best.pt --out outputs/demo_predictions \
  --num 20 --split test
```

Writes `*_input.png`, `*_target.png`, `*_prediction.png`, `*_diff.png`, a combined
`*_panel.png`, and raw predicted grids (`.npy` + `.json`). Add `--export-blueprint`
for an experimental blueprint-JSON export of each prediction.

## 8. Current results

Trained on **320 dense real blueprints** (224 train / 48 val / 48 test, split by
blueprint) extracted from 10 FactorioBin books. Held-out **test** set, best
checkpoint (~2.27 M-param U-Net, 24 epochs, ~42 min on Apple MPS):

| Metric (masked cells) | **Model** | always-EMPTY | always-majority |
|---|---:|---:|---:|
| non_empty_f1 (detection) | **0.633** | 0.000 | 0.366 |
| entity_token_acc (exact name+dir) | **0.532** | 0.000 | 0.077 |
| top5_acc | **0.962** | — | — |
| masked_acc | 0.725 | 0.776 | 0.017 |

**The model beats both trivial baselines decisively** — exact entity-token accuracy
is ≈6.9× the majority baseline, and the correct token is in the top-5 96% of the
time. It visibly reconstructs Factorio structure: belt lanes, electric-pole
corridors, and beacon grids continue straight through the masked hole (see
`outputs/demo_predictions/*_panel.png`). `masked_acc` dips just below always-EMPTY
because the model actually places entities (the intended precision/recall tradeoff).

See [`docs/findings.md`](docs/findings.md) for the full write-up (corpus, vocab,
what was learned, what failed, next steps) and [`cloud/README.md`](cloud/README.md)
to run training on a cloud GPU.

**How to read the metrics** (all computed over masked cells only):

- `masked_acc` — fraction of masked cells predicted exactly right. Misleading on
  its own: EMPTY is ~73% of masked cells, so always-EMPTY already scores ~0.73.
- `non_empty_f1` — detection F1 for "is there an entity here?" (pred non-empty vs.
  target non-empty). The headline *did-it-find-structure* metric.
- `entity_token_acc` — among masked cells that truly contain an entity, the
  fraction where the exact `name|direction` token is correct. The *did-it-get-the-
  -right-thing* metric.
- `top5_acc` — target token within the model's top-5 per cell.
- Baselines: `always-EMPTY` (non_empty_f1 = 0) and `always-majority-entity`
  (predict the single most common entity token).

**Success = the model beats both trivial baselines on non-empty reconstruction.**
Raw accuracy alone is not the goal.

## 9. Known limitations

- Anchors only: large entities occupy one cell, so dense multi-tile machines
  collide (collision rate is reported and is < 0.2% here).
- Mixing Factorio versions fragments direction tokens (1.1 uses 8-direction
  values, 2.0 uses 16). The seed list is intentionally all 1.1.
- Blueprints smaller than the 16×16 mask are dropped (`--min-dim`) so every
  example keeps real context around the hole.
- Small corpus (hundreds of blueprints). Patches are sampled on the fly, so
  effective example count is large, but blueprint diversity is limited.

## 10. Next steps

- Boundary conditioning (mask the interior but expose IO belts/pipes crossing the
  border) to make it "fill this hole usefully".
- Separate channels for name / direction / recipe / modules; model true entity
  footprints instead of anchors.
- Validity post-processing (legal directions, underground pairing, belt
  connectivity) and, eventually, Factorio-headless validation + throughput ranking.
- Graph representation (belt/pipe/rail adjacency, power coverage) alongside the
  grid.

---

## Repo layout

```
src/factorio_patches/
  download_factoriobin.py   # M1 polite downloader
  blueprint_decode.py       # M2 string -> JSON
  blueprint_extract.py      # M3 recursive book -> blueprints.jsonl
  vocab.py                  # token vocabulary (name|dir, EMPTY/MASK/UNK)
  rasterize.py              # M4 blueprint -> int token grid + stats
  dataset.py                # M5 masked-patch dataset + train/val/test split
  render.py                 # M6 grid/prediction PNGs
  model.py                  # M7 embedding + U-Net
  metrics.py                # M8 masked metrics + baselines + loss
  train.py                  # training loop
  eval.py                   # M9 prediction demo + raw export
scripts/   make_tiny_dataset.py, train_tiny.py, render_predictions.py
tests/     test_blueprint_decode.py, test_rasterize.py, test_masking.py
cloud/     Dockerfile, entrypoint.sh, run_local_gpu.sh, README.md (GCP/AWS/Nebius runbooks)
docs/      findings.md
```

## Data provenance & politeness

Seed URLs were **discovered** (web search + FactorioBin's public API), not
brute-forced from sequential post IDs. The downloader sends a descriptive
User-Agent, sleeps 2–5 s between requests, caches everything, and never
re-downloads a cached blueprint. Backup bulk sources (FactorioPrints' Firebase
DB, the Factorio School API, and GitHub raw `.txt` collections) are noted in
`docs/findings.md` if more data is needed.
