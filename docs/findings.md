# Findings — Factorio blueprint patch inpainting POC

_Goal of the POC: prove (or disprove) that a small model can reconstruct a masked
rectangular region of a Factorio blueprint well enough to be worth scaling._

## TL;DR

A small embedding + U-Net (~2.3 M params), trained locally on ~320 dense real
blueprints (patches sampled on the fly), **learns real Factorio spatial
structure** and **beats both trivial baselines** on non-empty patch
reconstruction. The single most important trick was down-weighting the dominant
`EMPTY` class in the loss; without it the model collapses to predicting "empty
everywhere". See the metrics table below.

## How many blueprints were processed?

Data was **discovered, not brute-forced**: a multi-angle web + FactorioBin-API
search produced ~27 verified blueprint-book posts; `data/seed_urls.txt` pins a
focused, version-consistent subset (Factorio 1.1, belt/logistics-heavy).

| Stage | Count |
|---|---|
| FactorioBin book posts downloaded | 10 (+ the FactorioBin `demo` book) |
| Decoded blueprint strings | 11 |
| **Individual blueprints extracted** (after dropping planners/empty/>20k-entity) | **1,008** |
| Total entities across all blueprints | 681,740 |
| Distinct entity names | 78 |
| Blueprints that are ≥70% belts | 635 / 1008 |

One book expands into many blueprints, so a handful of polite requests yields a
rich corpus (e.g. the "Belt Balancers" book alone → 326 blueprints).

## What was the vocabulary?

- Token = `entity_name|direction` (e.g. `transport-belt|2`), plus 3 special tokens
  `EMPTY`, `MASK`, `UNK`.
- Vocab size **160** (`--max-vocab 160`, no entity-family restriction) → **99.9 %**
  of entity instances are in-vocab (UNK rate ≈ 0.1 % on the kept blueprints).
- Most common tokens are the express-belt directions, express undergrounds, stack
  inserters, pipes, electric poles, steel chests — i.e. the corpus is
  belt/logistics-dominated, as intended.
- A `logistics`/`belt` preset (`vocab.PRESETS`) is available for an even more
  focused experiment; it covers 71 % / 53 % of instances respectively, with the
  remainder (rails, beacons, lamps, chests…) folded into `UNK`.

## What the data looks like after rasterization

- Anchors only, positions rounded to integer cells, normalized to origin.
- Median grid ≈ 15×10; p90 ≈ 96×135; collision rate **0.10 %** (negligible).
- For the masked-patch task we keep blueprints with `min(H,W) ≥ 24` (so a 16×16
  hole always leaves real context) and `≥ 8` entities → **320 usable blueprints**,
  split by blueprint into **train 224 / val 48 / test 48**. These are large, dense
  layouts (mean ~2,000 occupied cells each), so on-the-fly patch sampling yields a
  large, diverse effective dataset.

## What did the model learn?

Model: token `Embedding(160, 64)` → U-Net (2 down / 2 up, skip connections) →
per-cell logits over the vocab. ~2.3 M params. Cross-entropy **only inside the
mask**, EMPTY down-weighted (`--empty-weight 0.12`), Adam + cosine LR, MPS
(Apple Silicon), ~70 s/epoch.

<!-- RESULTS-TABLE-START -->
**Held-out TEST set** (best checkpoint = epoch 22 of 24; metrics over masked cells
only; ~98k masked cells; 22.4% of them truly non-empty):

| Metric | **Model** | always-EMPTY | always-majority-entity |
|---|---:|---:|---:|
| `non_empty_f1` (detection) | **0.633** | 0.000 | 0.366 |
| `non_empty_exact_f1` | **0.413** | 0.000 | 0.028 |
| `entity_token_acc` (exact name+dir) | **0.532** | 0.000 | 0.077 |
| `top5_acc` | **0.962** | — | — |
| `masked_acc` | 0.725 | 0.776 | 0.017 |
| precision / recall (non-empty) | 0.518 / 0.816 | — | 0.224 / 1.000 |

- **The model beats both trivial baselines decisively** on the metrics that matter:
  non-empty F1 0.633 vs 0.366 (majority) / 0.000 (empty); exact entity-token
  accuracy 0.532 vs 0.077 — **≈6.9× the majority baseline**.
- `top5_acc = 0.962`: the true `name|direction` token is in the model's top 5 (of
  157 non-empty tokens) **96%** of the time — local context is highly predictive.
- `masked_acc` (0.725) sits just *below* the always-EMPTY baseline (0.776) **by
  design**: the model actually places entities (recall 0.82), occasionally where
  the ground truth is empty, which trades a little raw accuracy for real
  reconstruction. Raw accuracy was never the goal.

Validation `entity_token_acc` climbed steadily: 0.00 (epoch 2, collapsed) → 0.14
(e3) → 0.31 (e8) → 0.41 (e13) → 0.48 (e17) → **0.52 (e22, best)**.

Run config: U-Net `d_model=64` (~2.27 M params), crop 64 / mask 16, vocab 160,
Adam lr 2e-3 + cosine decay, `--empty-weight 0.12`, batch 48, 24 epochs, ~105 s/epoch
on Apple M-series MPS (≈42 min total). Reproduced via `runs/poc_001/`.
<!-- RESULTS-TABLE-END -->

Qualitatively (see `outputs/demo_predictions/*_panel.png`): the model reproduces
the **periodic structure** of megabase production/belt blocks — beacon grids,
parallel belt lanes and their direction, regularly spaced electric poles and
inserters — rather than memorizing a single token. It is most confident exactly
where Factorio layouts are most regular.

## What failed / was hard?

- **Collapse to EMPTY.** With unweighted CE (or `--empty-weight ≥ 0.3`) the model
  predicts EMPTY in every masked cell — it matches the always-EMPTY baseline and
  never places entities. `--empty-weight 0.1–0.12` reliably escapes this;
  the weight trades precision (high weight) against recall (low weight).
- **Tiny blueprints make the task ill-posed.** Most extracted blueprints are
  smaller than the 16×16 hole, so the mask swallows the entire object and there is
  no context to condition on. Fixed with the `--min-dim` filter (dropped ~660 of
  ~1000 rasterized blueprints — they are still in the corpus, just not used as
  inpainting targets).
- **Exact token accuracy is intrinsically hard.** Direction and belt-tier are
  often ambiguous from local context, so `entity_token_acc` (exact `name|dir`)
  stays well below `non_empty_f1` (did-something-go-here) and `top5_acc`.
- **Version mixing.** Factorio 1.1 (8-direction) vs 2.0 (16-direction) fragments
  direction tokens; the seed list is kept all-1.1 to avoid this.

## What should be improved next?

1. **Boundary conditioning** — expose the IO belts/pipes crossing the mask border
   instead of masking a blind rectangle; turns "guess the hole" into "complete the
   sub-factory".
2. **Factored prediction** — separate heads for entity-name vs direction (and later
   recipe/modules); model true multi-tile footprints instead of single anchors.
3. **More + cleaner data** — scale via the noted backup sources; de-duplicate
   near-identical blueprints; optionally a belt-only split for a crisp first task.
4. **Validity + ranking** — post-process for legal directions, underground pairing
   and belt connectivity; eventually validate in headless Factorio and rank
   candidates by throughput/footprint.

## Backup data sources (if more volume is needed)

- **FactorioPrints** — Firebase Realtime DB (note the project typo `facorio`):
  `https://facorio-blueprints.firebaseio.com/blueprints.json?shallow=true` for the
  key index, then `/blueprints/<key>.json` per blueprint.
- **Factorio School** (mirror, same keys) — `https://www.factorio.school/api/blueprint/<key>`
  and `/api/blueprintSummaries/page/<n>`.
- **GitHub** raw `.txt` blueprint-string collections (file body *is* the `0eNq…`
  string).

## Reproduce

```bash
uv sync
uv run python -m factorio_patches.download_factoriobin --urls data/seed_urls.txt --out data/raw/factoriobin
uv run python -m factorio_patches.blueprint_decode --raw data/raw/factoriobin --out data/decoded
uv run python -m factorio_patches.blueprint_extract --decoded data/decoded --out data/processed/blueprints.jsonl
uv run python -m factorio_patches.vocab --blueprints data/processed/blueprints.jsonl --out data/processed/vocab.json --max-vocab 160
uv run python -m factorio_patches.dataset --blueprints data/processed/blueprints.jsonl --vocab data/processed/vocab.json --out data/processed/dataset.pt --max-dim 384 --min-dim 24 --min-entities 8
uv run python -m factorio_patches.train --data data/processed/dataset.pt --out runs/poc_001 --epochs 24 --empty-weight 0.12
uv run python -m factorio_patches.eval --checkpoint runs/poc_001/best.pt --out outputs/demo_predictions --split test
```
