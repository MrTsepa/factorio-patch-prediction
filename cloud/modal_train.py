"""Train the inpainting models on a Modal GPU, logging to Weights & Biases.

Runs the OLD (U-Net) and NEW (transformer) architectures on the SAME dataset in
parallel GPU containers, so they're directly comparable in one wandb project.

    uv run modal run cloud/modal_train.py --epochs 1 --samples 384   # smoke test
    uv run modal run cloud/modal_train.py                            # full comparison

Prereqs: `modal setup` (auth) + a Modal secret named 'wandb' (WANDB_API_KEY).
"""

import modal

app = modal.App("factorio-patch-inpaint")

# torch's default Linux wheels are CUDA-enabled; Modal provides the GPU + driver.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy", "pillow", "matplotlib", "requests",
                 "tqdm", "scikit-image", "wandb")
    .add_local_python_source("factorio_patches")
    .add_local_file("data/processed/dataset20.pt", "/root/dataset.pt")
)

# persists checkpoints + per-epoch prediction PNGs across runs
vol = modal.Volume.from_name("factorio-runs", create_if_missing=True)

# old vs new, same data. d_model differs by arch; lr tuned per arch.
# patch=2 transformer (1024 tokens) is memory-heavy: batch 64 fits A10G (24GB) with
# bf16 AMP; the U-Net is light so it gets a bigger batch.
CONFIGS = [
    ("unet", "unet-d96", dict(d_model=96, lr=2e-3, batch_size=128)),
    ("transformer", "tf-p2-d256",
     dict(d_model=256, depth=8, heads=8, patch=2, lr=1e-3, batch_size=64)),
]


@app.function(image=image, gpu="A10G", timeout=7200,
              secrets=[modal.Secret.from_name("wandb")],
              volumes={"/runs": vol})
def train_remote(arch: str, run_name: str, hp: dict, epochs: int, samples: int):
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
        train_samples=samples, val_samples=1024,
        num_workers=2, device="auto", seed=0, amp="auto", compile=True,
        wandb="factorio-patch-inpaint", run_name=run_name)
    summary = train(args)
    vol.commit()
    test = (summary.get("test") or {}).get("model", {})
    return {"run": run_name, "arch": arch, "params": summary.get("model_params"),
            "best_val_entity_acc": summary.get("best_val_entity_token_acc"),
            "test_entity_acc": test.get("entity_token_acc"),
            "test_nonempty_f1": test.get("non_empty_f1"),
            "test_top5": test.get("top5_acc")}


@app.local_entrypoint()
def main(epochs: int = 30, samples: int = 8000):
    handles = [train_remote.spawn(arch, name, hp, epochs, samples)
               for arch, name, hp in CONFIGS]
    print(f"launched {len(handles)} runs on Modal A10G (epochs={epochs}, samples={samples})")
    for h in handles:
        print("  ->", h.get())
