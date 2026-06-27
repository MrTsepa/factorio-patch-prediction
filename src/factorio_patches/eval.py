"""Evaluate a trained checkpoint: metrics + before/target/prediction/diff demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import load_dataset, make_datasets
from .metrics import evaluate, format_metrics, most_common_entity_id
from .model import PatchInpaintUNet
from .render import render_prediction_set
from .vocab import EMPTY_ID, MASK_ID, Vocab, split_token
from torch.utils.data import DataLoader


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocab(ckpt["vocab_tokens"])
    model = PatchInpaintUNet(len(vocab), d_model=ckpt["config"]["d_model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt


def grid_to_blueprint(grid: np.ndarray, vocab: Vocab) -> dict:
    """EXPERIMENTAL: convert a predicted token grid back into a minimal blueprint dict.

    Not Factorio-perfect (anchors only, no collision/validity checks).
    """
    entities = []
    n = 0
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            tid = int(grid[r, c])
            if tid in (EMPTY_ID, MASK_ID):
                continue
            tok = vocab.decode(tid)
            name, direction = split_token(tok)
            if name in ("UNK",):
                continue
            n += 1
            ent = {"entity_number": n, "name": name,
                   "position": {"x": float(c), "y": float(r)}}
            if direction:
                ent["direction"] = int(direction)
            entities.append(ent)
    return {"blueprint": {"item": "blueprint", "entities": entities, "version": 0,
                          "label": "EXPERIMENTAL predicted patch"}}


def run_demo(args) -> dict:
    device = torch.device(args.device) if args.device != "auto" else (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu"))
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)

    ds_path = Path(args.data) if args.data else Path(ckpt["dataset"])
    payload = load_dataset(ds_path)
    ds = make_datasets(payload)
    split_ds = ds[args.split]
    if split_ds is None:
        print(f"split '{args.split}' is empty")
        return {}

    prior_id = ckpt.get("prior_id") or most_common_entity_id(payload["splits"]["train"])
    loader = DataLoader(split_ds, batch_size=64, shuffle=False)
    metrics = evaluate(model, loader, device, prior_id)
    print(format_metrics(args.split, metrics))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    grids_dir = out / "grids"
    grids_dir.mkdir(exist_ok=True)

    n = min(args.num, len(split_ds))
    with torch.no_grad():
        for i in range(n):
            s = split_ds[i]
            x = s["x"].unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            y = s["y"].numpy()
            mask = s["mask"].numpy()
            xin = s["x"].numpy()
            pred_full = y.copy()
            pred_full[mask] = pred[mask]
            render_prediction_set(xin, y, pred_full, mask, vocab, out, f"{args.split}_{i:03d}",
                                  cell=args.cell)
            np.save(grids_dir / f"{args.split}_{i:03d}_pred.npy", pred_full)
            rec = {"input": xin.tolist(), "target": y.tolist(),
                   "prediction": pred_full.tolist(), "mask": mask.astype(int).tolist()}
            (grids_dir / f"{args.split}_{i:03d}.json").write_text(json.dumps(rec))
            if args.export_blueprint:
                bp = grid_to_blueprint(pred_full, vocab)
                (grids_dir / f"{args.split}_{i:03d}_blueprint.json").write_text(json.dumps(bp))

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {n} demo panel(s) + raw grids -> {out}")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint and render a prediction demo.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None, help="dataset.pt (defaults to the one in the checkpoint)")
    ap.add_argument("--out", type=Path, default=Path("outputs/demo_predictions"))
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--cell", type=int, default=14)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--export-blueprint", action="store_true",
                    help="also export experimental blueprint JSON from predictions")
    args = ap.parse_args(argv)
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
