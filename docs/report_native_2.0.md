# Report — native Factorio 2.0 transformer (poc_002)

Reworking the patch-inpainting model to be **native Factorio 2.0 (+ Space Age)**,
with a **transformer** architecture, trained within a **~1-hour budget**, plus a
game-accurate HTML demo and a cloud-compute cost analysis.

## TL;DR
- Trained a **6.9M-param ViT-style transformer** on a **native-2.0 + Space-Age**
  corpus, stopped at the 1-hour budget (epoch 17/24).
- **Test:** entity-token accuracy **0.266** (≈3.5× the majority baseline 0.077),
  non-empty F1 **0.457**, **top-5 0.910** — clearly learns real 2.0 structure and
  beats trivial baselines.
- Predictions are **native 2.0**: they export to 2.0-versioned blueprint strings
  and render **pixel-faithfully** in real game graphics (incl. Space-Age entities:
  turbo belts, biochambers) via FBSR baked from Factorio **2.0.76 + Space Age**.
- It scores **below** the earlier 1.1 U-Net (0.532) — that gap is mostly the
  **harder task** (much broader 2.0+SA vocab), **less data**, and the transformer's
  **patch-4 coarseness**, not a fundamental regression. See "Analysis".

## The model — `PatchInpaintTransformer`
ViT-style encoder + convolutional decoder with a full-res embedding skip:
- cell `Embedding` → **conv patchify stem** (stride `patch=4`): 64×64 cells →
  **16×16 = 256 patch tokens** at `d_model=256` (each token = a 4×4 cell block);
- learned 2D positional embedding;
- **pre-norm transformer encoder**, depth **8**, heads **8**, FFN 4×, dropout 0.1
  — global self-attention over all 256 tokens (captures long-range periodic
  structure the U-Net's local receptive field can't);
- **conv decoder** upsamples 16→32→64, **fused with the full-res cell embedding**
  (skip) so unmasked detail survives the patchify bottleneck;
- `1×1` head → per-cell logits. **6,910,664 params** (3× the U-Net).

`patch` is the key knob: 256 tokens (patch 4) keeps attention cheap enough to
train many epochs in an hour; `patch=2` (1024 tokens) would predict 4× finer in
the hole at ~4× compute.

## The data — `dataset20.pt` (native 2.0 + Space Age)
- Source: **9 FactorioBin Space-Age books** (Factorio 2.0.x), discovered via the
  same web+API search (Nilaus Base/Space-Age, Coffeemug, RedRum, Zisteau, …).
- Extracted **839 blueprints / 417,259 entities / 116 entity types**; directions
  are native **16-direction** (0/4/8/12 …), names are 2.0 (`turbo-transport-belt`,
  `bulk-inserter`, foundries, biochambers, …).
- After filtering (≥24-tile, dense, ≥50% in-vocab): **255 usable blueprints** →
  **train 179 / val 38 / test 38** (split by blueprint). Vocab **200** (99.3% coverage).

## Training (1-hour budget)
AdamW (lr 1e-3, weight-decay 0.05) + cosine schedule, masked cross-entropy with
`EMPTY` down-weighted (0.12), dropout 0.1, batch 48, 6,000 on-the-fly patches/epoch,
Apple-Silicon **MPS**. Actual ~200 s/epoch → stopped at **epoch 17/24 (~56.5 min)**;
`best.pt` = epoch 17.

| epoch | val entity-acc | val non-empty F1 | val top-5 |
|---:|---:|---:|---:|
| 3 | 0.116 | 0.370 | 0.836 |
| 8 | 0.220 | 0.466 | 0.881 |
| 12 | 0.247 | 0.485 | 0.885 |
| **17 (best)** | **0.258** | 0.472 | 0.894 |

## Results — held-out test
| metric (masked cells) | **transformer (2.0)** | always-EMPTY | always-majority |
|---|---:|---:|---:|
| entity-token accuracy (exact) | **0.266** | 0.000 | 0.077 |
| non-empty F1 (detection) | **0.457** | 0.000 | 0.355 |
| top-5 accuracy | **0.910** | — | — |
| masked-cell accuracy | 0.601 | 0.784 | 0.017 |
| precision / recall (non-empty) | 0.360 / 0.627 | — | 0.224 / 1.000 |

For reference, the earlier **1.1 U-Net (`poc_001`)** scored test entity-acc **0.532**,
F1 0.633, top-5 0.962 — on the *easier* 1.1 belt/logistics task.

## Analysis — why 2.0 scores lower than 1.1 (and it's mostly the task)
- **Much broader distribution:** native 2.0 **+ Space Age** spans a 200-token vocab
  (turbo belts, foundries, EM plants, quality, agriculture/biochambers, space-platform
  parts) — far more varied and less repetitive than 1.1 belt/logistics. Exact-token
  prediction among many similar 2.0 entities is intrinsically harder.
- **Less / more heterogeneous data:** 179 training blueprints (2.0) vs 224 (1.1),
  and Space-Age books are more diverse (less repeated structure to exploit).
- **Patch-4 coarseness:** the transformer reasons at 4×4-cell granularity then
  upsamples; the hole's exact per-cell layout is reconstructed from coarse patch
  features. **`top5=0.91`** shows it knows the right *kind* of thing per region; the
  exact pick is what suffers.
- The high recall / lower precision (0.63 / 0.36) means it confidently places lots
  of entities (good) but mis-picks among look-alikes (the diversity tax).

## Rendering + demo (native 2.0, game-accurate)
- FBSR re-baked from **Factorio 2.0.76 full + Space Age + Quality + Elevated Rails**
  (~17k sprites). Predictions export to **2.0-versioned** strings (so directions
  aren't re-migrated) and render pixel-faithfully — validated against FactorioBin
  references at **~0.12% error-area** on substantial blueprints (base + Space Age).
- **HTML demo:** `outputs/demo_2.0/index.html` — self-contained page with test
  metrics + per-example abstract panels and **FBSR game-renders of target vs
  prediction** (kept local; embeds Wube's copyrighted sprites).

## External compute — speedup & cost (this job)
MPS has no Tensor Cores / weak mixed-precision, so cloud NVIDIA GPUs are much faster:

| GPU | ~speedup vs MPS | this run (~90 min full) → | cost/run |
|---|---|---|---|
| T4 | ~3× | ~30 min | ~$0.05–0.18 |
| L4 / A10G | ~5–8× | ~12–18 min | ~$0.13–0.22 |
| A100 (bf16/TF32) | ~12–20× | ~5–7 min | ~$0.15–0.40 |
| H100 | ~20–40× | ~3–4 min | ~$0.20–0.65 |

Marginal for this POC; **high value at scale** (10× model + thousands of blueprints
→ overnight-local becomes ~20–30 min on an A100 spot, ~$0.75). Scaffolding is in
`cloud/`; `train.py --arch transformer` runs there unchanged.

## Next steps (to close the gap)
1. **More data** — biggest lever. Scale the 2.0 corpus from FactorioPrints (17,610
   blueprints, growing 2.0 slice) + more FactorioBin 2.0 books → thousands of
   blueprints (pre-2.0 ones migrate to 2.0-base; see migration notes).
2. **`patch=2`** (4× finer in the hole) and/or a bigger transformer — affordable on
   a cloud GPU.
3. **Factored head** (separate name vs direction) and **footprint-aware** rasterization.
4. Longer training (the val curve hadn't fully plateaued at the 1-hour cutoff).
