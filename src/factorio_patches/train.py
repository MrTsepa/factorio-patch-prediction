"""Train the U-Net patch-inpainting model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_dataset, load_dataset, make_datasets
from .metrics import evaluate, format_metrics, masked_ce_loss, most_common_entity_id
from .model import PatchInpaintUNet
from .render import render_prediction_set
from .vocab import EMPTY_ID, Vocab


def pick_device(arg: str) -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset(data: Path, crop: int, mask: int, max_dim: int, seed: int) -> Path:
    """Return a dataset.pt path, building it from blueprints.jsonl + vocab.json if needed."""
    data = Path(data)
    if data.is_file():
        return data
    ds_path = data / "dataset.pt"
    if ds_path.exists():
        return ds_path
    bp = data / "blueprints.jsonl"
    vp = data / "vocab.json"
    if not (bp.exists() and vp.exists()):
        raise FileNotFoundError(
            f"no dataset.pt and missing {bp} or {vp}; run extract + vocab first")
    print(f"dataset.pt not found; building from {bp} + {vp} ...")
    build_dataset(bp, Vocab.load(vp), ds_path, crop_size=crop, mask_size=mask,
                  max_dim=max_dim, seed=seed)
    return ds_path


def render_epoch_samples(model, val_ds, vocab, device, out_dir: Path, n: int = 6):
    if val_ds is None:
        return
    model.eval()
    n = min(n, len(val_ds))
    with torch.no_grad():
        for i in range(n):
            s = val_ds[i]
            x = s["x"].unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            y = s["y"].numpy()
            mask = s["mask"].numpy()
            xin = s["x"].numpy()
            # show ground truth tokens outside the mask in the prediction panel
            pred_full = y.copy()
            pred_full[mask] = pred[mask]
            render_prediction_set(xin, y, pred_full, mask, vocab, out_dir, f"val_{i:02d}")


def train(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_path = resolve_dataset(Path(args.data), args.crop_size, args.mask_size, args.max_dim, args.seed)
    payload = load_dataset(ds_path)
    cfg = payload["config"]
    # Honor the dataset's crop/mask (the model is conv so it adapts, but keep them aligned).
    ds = make_datasets(payload, train_length=args.train_samples, val_length=args.val_samples, seed=args.seed)
    vocab = ds["vocab"]
    prior_id = most_common_entity_id(payload["splits"]["train"])

    print(f"device={device} vocab={len(vocab)} crop={cfg['crop_size']} mask={cfg['mask_size']}")
    print(f"split sizes: train={len(payload['splits']['train'])} "
          f"val={len(payload['splits']['val'])} test={len(payload['splits']['test'])} blueprints")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers) if ds["val"] else None
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if ds["test"] else None

    model = PatchInpaintUNet(len(vocab), d_model=args.d_model).to(device)
    print(f"model params: {model.num_params():,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # Down-weight EMPTY in the loss to fight the heavy class imbalance.
    class_weight = torch.ones(len(vocab), device=device)
    class_weight[EMPTY_ID] = args.empty_weight

    history = []
    best_metric = -1.0
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        nb = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            logits = model(x)
            loss = masked_ce_loss(logits, y, mask, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
            pbar.set_postfix(loss=f"{running / nb:.3f}")
        train_loss = running / max(nb, 1)
        sched.step()

        val_metrics = evaluate(model, val_loader, device, prior_id) if val_loader else None
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss, "seconds": round(dt, 1),
               "val": val_metrics}
        history.append(rec)
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"\nepoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  ({dt:.1f}s)")
        if val_metrics:
            print(format_metrics("val", val_metrics))

        # checkpoints
        ckpt = {"model_state": model.state_dict(), "vocab_tokens": vocab.itos,
                "config": {"crop_size": cfg["crop_size"], "mask_size": cfg["mask_size"],
                           "d_model": args.d_model}, "epoch": epoch,
                "val_metrics": val_metrics, "prior_id": prior_id, "dataset": str(ds_path)}
        torch.save(ckpt, out / "last.pt")
        crit = val_metrics["model"]["entity_token_acc"] if val_metrics else -1.0
        if crit > best_metric:
            best_metric = crit
            torch.save(ckpt, out / "best.pt")
            print(f"  -> new best (val entity_token_acc={crit:.3f}) saved to best.pt")

        render_epoch_samples(model, ds["val"], vocab, device, out / "preds" / f"epoch_{epoch:02d}", n=6)

    # Final test evaluation with the best checkpoint.
    summary = {"best_val_entity_token_acc": best_metric, "epochs": args.epochs,
               "device": str(device), "model_params": model.num_params(),
               "empty_weight": args.empty_weight, "lr": args.lr, "d_model": args.d_model,
               "crop_size": cfg["crop_size"], "mask_size": cfg["mask_size"]}
    if test_loader:
        best_path = out / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, weights_only=False)
            model.load_state_dict(best["model_state"])
            summary["best_epoch"] = best["epoch"]
        else:
            print("\n[warn] no best.pt (no validation split?); evaluating test with last model")
        test_metrics = evaluate(model, test_loader, device, prior_id)
        print("\n=== TEST (best checkpoint) ===")
        print(format_metrics("test", test_metrics))
        summary["test"] = test_metrics
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved checkpoints + metrics to {out}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the patch-inpainting U-Net.")
    ap.add_argument("--data", type=Path, default=Path("data/processed"),
                    help="dir with blueprints.jsonl+vocab.json (auto-builds dataset.pt) or a dataset.pt file")
    ap.add_argument("--out", type=Path, default=Path("runs/poc_001"))
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--mask-size", type=int, default=16)
    ap.add_argument("--max-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--empty-weight", type=float, default=0.2,
                    help="loss weight for the EMPTY class (down-weight to fight imbalance)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--train-samples", type=int, default=4000, help="patches sampled per epoch")
    ap.add_argument("--val-samples", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
