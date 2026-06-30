"""Train the inpainting models on a Modal GPU, logging to Weights & Biases.

Runs the OLD (U-Net) and NEW (transformer) architectures on the SAME dataset in
parallel GPU containers, so they're directly comparable in one wandb project.
Both are early-stopped (val entity_token_acc, patience) and hard-capped to a
<=3h wall-clock budget, so neither container ever overruns.

    # smoke test (tiny, A10G both):
    uv run modal run cloud/modal_train.py --epochs 2 --samples 384 --smoke
    # full 3h comparison on the scaled dataset:
    uv run modal run cloud/modal_train.py

Prereqs: `modal setup` (auth) + a Modal secret named 'wandb' (WANDB_API_KEY),
and the scaled dataset built locally at LOCAL_DATASET (see below).
"""

import os
import time
from pathlib import Path

import modal

app = modal.App("factorio-patch-inpaint")

# ---------------------------------------------------------------------------- #
# Data: BAKE the dataset into the image. The scaled (~5000-blueprint) dataset is
# still small (tens of MB), immutable, and baking guarantees both arch containers
# train on byte-identical data -- essential for a fair old-vs-new comparison.
# Build it locally first, e.g.:
#   uv run python -m factorio_patches.dataset \
#       --blueprints data/processed/blueprints5k.jsonl --vocab data/processed/vocab5k.json \
#       --out data/processed/dataset5k.pt --max-dim 384 --min-dim 24 \
#       --min-entities 8 --min-invocab-frac 0.5
# (If the dataset ever grows past ~1GB or you iterate on it between runs, switch
#  to a Modal Volume instead -- see the commented alternative at the bottom.)
# ---------------------------------------------------------------------------- #
# Prefer the scaled dataset; fall back to dataset20.pt so smoke runs work today.
# NOTE: this module is imported in the CONTAINER too (to hydrate the function), where the
# local data/ paths don't exist -- so a bare next() would raise StopIteration and crash the
# container at import. The default keeps import safe; the container reads /root/dataset.pt.
LOCAL_DATASET = next((p for p in ("data/processed/dataset5k.pt",
                                  "data/processed/dataset20.pt")
                      if os.path.exists(p)), "data/processed/dataset5k.pt")
DATA_TAG = Path(LOCAL_DATASET).stem.replace("dataset", "") or "ds"   # e.g. "5k"
print(f"[modal_train] baking dataset: {LOCAL_DATASET}")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pillow", "matplotlib", "requests",
                 "tqdm", "scikit-image", "wandb")
    .add_local_python_source("factorio_patches")
    .add_local_file(LOCAL_DATASET, "/root/dataset.pt")
)

# persists checkpoints (best.pt/last.pt) + per-epoch prediction PNGs across runs
vol = modal.Volume.from_name("factorio-runs", create_if_missing=True)

# ---------------------------------------------------------------------------- #
# Budget: 3h hard cap. Modal `timeout` is the absolute kill switch; the in-process
# --max-seconds stops BEFORE an epoch that would overrun, leaving headroom for the
# final best-checkpoint test eval + vol.commit.
# ---------------------------------------------------------------------------- #
WALL_HARD_S = 10800      # 3h -- Modal container timeout (absolute backstop)
WALL_SOFT_S = 10200      # 2h50m -- in-loop cap; ~10min reserved for test eval/commit

# old vs new, SAME data. Per-arch GPU: the U-Net is light (A10G is plenty); the
# patch=2 transformer (32x32 = 1024 tokens, d256/depth8) is attention-heavy and
# data-hungry -- give it an A100-40GB so it fits batch 128 and gets the most
# epochs inside the 3h budget (this 20x data is the real test of whether it
# catches up). bf16 AMP + torch.compile are on for both.
# Architecture comparison on identical leak-free data: the diagnosis said the patch=2 ViT
# fails on cell-precise placement (thin decoder + coarse patches), so we test U-Net-based
# variants — pure scale, bottleneck self-attention (UNETR-lite), and axial attention (the
# axis-aligned-line prior for belts/pipes) — all ~12M params for a size-fair comparison.
CONFIGS = [
    dict(arch="unet", name="unet-d96", gpu="A10G",                      # 5.1M baseline / bar
         hp=dict(d_model=96, lr=2e-3, batch_size=128, weight_decay=0.01)),
    dict(arch="unet-scaled", name="unet-scaled-d80", gpu="A10G",        # ~14M: does scale alone win?
         hp=dict(d_model=80, lr=2e-3, batch_size=128, weight_decay=0.01)),
    dict(arch="unet-attn", name="unet-attn-d96", gpu="A10G",            # ~12M: bottleneck attention
         hp=dict(d_model=96, depth=4, heads=8, lr=1.5e-3, batch_size=128, weight_decay=0.01)),
    dict(arch="unet-axial", name="unet-axial-d128", gpu="A10G",         # ~12M: axial attention
         hp=dict(d_model=128, heads=8, lr=1.5e-3, batch_size=96, weight_decay=0.01)),
]

# MaskGIT across DIFFERENT backbones (same data/budget as the arch run): does iterative
# confidence decoding (joint coherence) beat the single-shot numbers — and on which backbone?
# Compare each vs its single-shot counterpart (unet 0.550, transformer 0.289, axial 0.502).
CONFIGS_MASKGIT = [
    dict(arch="unet", name="mg-unet-d96", gpu="A10G",                   # winner: does MG add?
         hp=dict(d_model=96, lr=2e-3, batch_size=128, weight_decay=0.01)),
    dict(arch="unet-axial", name="mg-axial-d128", gpu="A10G",
         hp=dict(d_model=128, heads=8, lr=1.5e-3, batch_size=96, weight_decay=0.01)),
    dict(arch="transformer", name="mg-tf-p2-d256", gpu="A100-40GB",     # does MG rescue the ViT?
         hp=dict(d_model=256, depth=8, heads=8, patch=2, lr=1e-3, batch_size=128, weight_decay=0.05)),
    dict(arch="unet-scaled", name="mg-scaled-d80", gpu="A10G",
         hp=dict(d_model=80, lr=2e-3, batch_size=128, weight_decay=0.01)),
]


@app.function(image=image, gpu="A10G", timeout=WALL_HARD_S,
              secrets=[modal.Secret.from_name("wandb")],
              volumes={"/runs": vol})
def train_remote(arch: str, run_name: str, hp: dict, epochs: int, samples: int,
                 val_samples: int, patience: int, max_seconds: float,
                 group: str = None, tags: list = None,
                 maskgit: bool = False, maskgit_steps: int = 8):
    from argparse import Namespace
    from pathlib import Path

    import torch
    from factorio_patches.train import train
    print("GPU:", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

    args = Namespace(
        data=Path("/root/dataset.pt"), out=Path(f"/runs/{run_name}"),
        crop_size=64, mask_size=16, max_dim=384,
        epochs=epochs, batch_size=hp.get("batch_size", 64),
        lr=hp.get("lr", 1e-3), weight_decay=hp.get("weight_decay", 0.05),
        empty_weight=0.12, arch=arch,
        d_model=hp.get("d_model", 256), depth=hp.get("depth", 8),
        heads=hp.get("heads", 8), patch=hp.get("patch", 2),
        train_samples=samples, val_samples=val_samples,
        num_workers=4, device="auto", seed=0, amp="auto", compile=True,
        size_power=hp.get("size_power", 0.5),   # mild size-weighting
        maskgit=maskgit, maskgit_steps=maskgit_steps,
        aug=hp.get("aug", False), label_smoothing=hp.get("label_smoothing", 0.0),
        # --- early stopping + 3h wall-clock budget ---
        patience=patience, min_delta=1e-3, max_seconds=max_seconds, restore_best=True,
        wandb="factorio-patch-inpaint", run_name=run_name,
        wandb_group=group, wandb_tags=tags)
    summary = train(args, on_epoch_end=vol.commit)   # persist checkpoints each epoch
    vol.commit()
    test = (summary.get("test") or {}).get("model", {})
    test_ss = (summary.get("test") or {}).get("model_singleshot", {})   # MaskGIT 2x2
    return {"run": run_name, "arch": arch, "params": summary.get("model_params"),
            "test_singleshot_entity_acc": test_ss.get("entity_token_acc"),
            "epochs_run": summary.get("epochs_run"), "best_epoch": summary.get("best_epoch"),
            "stopped_early": summary.get("stopped_early"),
            "wall_clock_s": summary.get("wall_clock_s"),
            "best_val_entity_acc": summary.get("best_val_entity_token_acc"),
            "test_entity_acc": test.get("entity_token_acc"),
            "test_nonempty_f1": test.get("non_empty_f1"),
            "test_io_acc": test.get("io_acc"),
            "test_top5": test.get("top5_acc")}


# AutoResearch Phase-B: short A10G proxy runs testing training-time levers (D4 augmentation,
# label smoothing, scale-with-aug). Edited between rounds as the greedy search proceeds.
CONFIGS_AUTO = [
    # Round 2: aug at 80ep already beat baseline WITH TTA (undertrained) -> train it LONG;
    # also test aug+capacity (scale alone was a wash, but aug may unlock it).
    dict(arch="unet", name="ar-auglong", gpu="A10G",
         hp=dict(d_model=96, lr=2e-3, batch_size=128, aug=True)),
    dict(arch="unet", name="ar-augbig", gpu="A10G",
         hp=dict(d_model=128, lr=2e-3, batch_size=128, aug=True)),
]


@app.local_entrypoint()
def main(epochs: int = 120, samples: int = 16000, val_samples: int = 4096,
         patience: int = 10, smoke: bool = False, maskgit: bool = False, auto: bool = False):
    """Launch a parallel comparison, each 3h-capped. --maskgit / --auto swap the config set."""
    max_seconds = 120.0 if smoke else float(WALL_SOFT_S)
    configs = CONFIGS_AUTO if auto else (CONFIGS_MASKGIT if maskgit else CONFIGS)
    # Unique + descriptive wandb names: <arch-config>-<data>-[smoke-]<MMDD-HHMM>, all
    # runs of one launch share a group so they overlay/compare cleanly (no more dupes).
    launch = ("smoke-" if smoke else "") + time.strftime("%m%d-%H%M")
    group = f"{DATA_TAG}-{'mg-' if maskgit else ''}{launch}"
    handles = []
    for c in configs:
        gpu = c["gpu"]                               # smoke validates the REAL per-arch GPU
        run_name = f"{c['name']}-{DATA_TAG}-{launch}"
        tags = [c["arch"], f"data:{DATA_TAG}", f"gpu:{gpu}"] + \
            (["maskgit"] if maskgit else []) + (["smoke"] if smoke else [])
        fn = train_remote.with_options(gpu=gpu, timeout=WALL_HARD_S)
        handles.append((run_name, fn.spawn(
            c["arch"], run_name, c["hp"], epochs, samples,
            val_samples, patience, max_seconds, group, tags, maskgit)))
    print(f"launched {len(handles)} runs (maskgit={maskgit}, epochs<={epochs}, "
          f"samples={samples}, patience={patience}, wall-cap={max_seconds:.0f}s)")
    for name, h in handles:
        print(f"  [{name}] ->", h.get())


@app.function(image=image, gpu="A10G", timeout=2400, volumes={"/runs": vol})
def eval_remote(configs, n=4000):
    """Evaluate ensemble/TTA configs on the GPU (entity-token accuracy on val + test)."""
    from pathlib import Path

    import torch
    from factorio_patches.dataset import load_dataset
    from factorio_patches.ensemble import entity_acc
    from factorio_patches.eval import load_checkpoint
    dev = torch.device("cuda")
    p = load_dataset("/root/dataset.pt")
    val, test = p["splits"]["val"], p["splits"]["test"]
    cache = {}

    def get(rn):
        if rn not in cache:
            m, v, _ = load_checkpoint(Path(f"/runs/{rn}/best.pt"), dev)
            m.eval(); cache[rn] = (m, v)
        return cache[rn]

    out = []
    for c in configs:
        mods, vocab = [], None
        for rn in c["runs"]:
            m, vocab = get(rn); mods.append(m)
        va = entity_acc(mods, val, vocab, dev, tta=c.get("tta", False), n=n)
        ta = entity_acc(mods, test, vocab, dev, tta=c.get("tta", False), n=n)
        out.append({"name": c["name"], "val": round(va, 4), "test": round(ta, 4)})
    return out


@app.local_entrypoint()
def evals():
    """GPU eval of ensemble / TTA configs (no local compute). Edit the config list."""
    AR, PB = "-5k-0628-0326", "-5k-0629-2227"
    AL, AB = "ar-auglong-5k-0630-0011", "ar-augbig-5k-0630-0011"
    U, S, X = f"unet-d96{AR}", f"unet-scaled-d80{AR}", f"unet-axial-d128{AR}"
    LS, A80 = f"ar-ls{PB}", f"ar-aug{PB}"
    cfgs = [
        {"name": "baseline U-Net", "runs": [U]},
        {"name": "D4 TTA (no aug -- dead end)", "runs": [U], "tta": True},
        {"name": "label smoothing (dead end)", "runs": [LS]},
        {"name": "U-Net axial (dead end)", "runs": [X]},
        {"name": "ensemble x3 (arch)", "runs": [U, S, X]},
        {"name": "D4 aug 80ep (undertrained)", "runs": [A80]},
        {"name": "aug 80ep + TTA", "runs": [A80], "tta": True},
        {"name": "D4 aug converged (139ep)", "runs": [AL]},
        {"name": "aug + D4 TTA", "runs": [AL], "tta": True},
        {"name": "aug ensemble (long+big)", "runs": [AL, AB]},
        {"name": "aug ensemble + TTA", "runs": [AL, AB], "tta": True},
        {"name": "aug + non-aug ens + TTA (dead end)", "runs": [AL, AB, U, S], "tta": True},
    ]
    for r in eval_remote.remote(cfgs, n=10000):
        print(f"AR_EVAL {r['name']} | val={r['val']} test={r['test']}")


# ---------------------------------------------------------------------------- #
# Volume alternative (use instead of add_local_file if the dataset is large or
# changes often). Upload once with `modal volume put`, then read it at runtime:
#
#   data_vol = modal.Volume.from_name("factorio-data", create_if_missing=True)
#   # $ modal volume put factorio-data data/processed/dataset5k.pt /dataset.pt
#   @app.function(..., volumes={"/runs": vol, "/data": data_vol})
#   def train_remote(...):
#       args = Namespace(data=Path("/data/dataset.pt"), ...)
# ---------------------------------------------------------------------------- #
