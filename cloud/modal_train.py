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
CONFIGS = [
    dict(arch="unet", name="unet-d96", gpu="A10G",
         hp=dict(d_model=96, lr=2e-3, batch_size=192, weight_decay=0.01)),
    dict(arch="transformer", name="tf-p2-d256", gpu="A100-40GB",
         hp=dict(d_model=256, depth=8, heads=8, patch=2, lr=1e-3,
                 batch_size=128, weight_decay=0.05)),
]


@app.function(image=image, gpu="A10G", timeout=WALL_HARD_S,
              secrets=[modal.Secret.from_name("wandb")],
              volumes={"/runs": vol})
def train_remote(arch: str, run_name: str, hp: dict, epochs: int, samples: int,
                 val_samples: int, patience: int, max_seconds: float,
                 group: str = None, tags: list = None):
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
        # --- early stopping + 3h wall-clock budget ---
        patience=patience, min_delta=1e-3, max_seconds=max_seconds, restore_best=True,
        wandb="factorio-patch-inpaint", run_name=run_name,
        wandb_group=group, wandb_tags=tags)
    summary = train(args, on_epoch_end=vol.commit)   # persist checkpoints each epoch
    vol.commit()
    test = (summary.get("test") or {}).get("model", {})
    return {"run": run_name, "arch": arch, "params": summary.get("model_params"),
            "epochs_run": summary.get("epochs_run"), "best_epoch": summary.get("best_epoch"),
            "stopped_early": summary.get("stopped_early"),
            "wall_clock_s": summary.get("wall_clock_s"),
            "best_val_entity_acc": summary.get("best_val_entity_token_acc"),
            "test_entity_acc": test.get("entity_token_acc"),
            "test_nonempty_f1": test.get("non_empty_f1"),
            "test_io_acc": test.get("io_acc"),
            "test_top5": test.get("top5_acc")}


@app.local_entrypoint()
def main(epochs: int = 80, samples: int = 16000, val_samples: int = 4096,
         patience: int = 8, smoke: bool = False):
    """Launch U-Net (A10G) + transformer (A100-40GB) in parallel, each 3h-capped."""
    max_seconds = 120.0 if smoke else float(WALL_SOFT_S)
    # Unique + descriptive wandb names: <arch-config>-<data>-[smoke-]<MMDD-HHMM>, all
    # runs of one launch share a group so they overlay/compare cleanly (no more dupes).
    launch = ("smoke-" if smoke else "") + time.strftime("%m%d-%H%M")
    group = f"{DATA_TAG}-{launch}"
    handles = []
    for c in CONFIGS:
        gpu = c["gpu"]                               # smoke validates the REAL per-arch GPU
        run_name = f"{c['name']}-{DATA_TAG}-{launch}"
        tags = [c["arch"], f"data:{DATA_TAG}", f"gpu:{gpu}"] + (["smoke"] if smoke else [])
        fn = train_remote.with_options(gpu=gpu, timeout=WALL_HARD_S)
        handles.append((run_name, fn.spawn(
            c["arch"], run_name, c["hp"], epochs, samples,
            val_samples, patience, max_seconds, group, tags)))
    print(f"launched {len(handles)} runs (epochs<={epochs}, samples={samples}, "
          f"patience={patience}, wall-cap={max_seconds:.0f}s)")
    for name, h in handles:
        print(f"  [{name}] ->", h.get())


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
