"""Evaluate a trained checkpoint: metrics + before/target/prediction/diff demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .blueprint_decode import encode_blueprint_string
from .dataset import load_dataset, make_datasets
from .metrics import evaluate, format_metrics, most_common_entity_id
from .model import build_model
from .render import render_prediction_set
from .vocab import EMPTY_ID, MASK_ID, Vocab, split_token
from torch.utils.data import DataLoader


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, weights_only=False)
    vocab = Vocab(ckpt["vocab_tokens"])
    cfg = ckpt["config"]
    model = build_model(cfg.get("arch", "unet"), len(vocab), d_model=cfg["d_model"],
                        depth=cfg.get("depth", 6), heads=cfg.get("heads", 6),
                        patch=cfg.get("patch", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab, ckpt


# Factorio build numbers ((major<<48)|(minor<<32)|(patch<<16)|dev).
VERSION_1_1 = 281479275151360   # 1.1.x  (8-direction)
VERSION_2_0 = 562949953421312   # 2.0.x  (16-direction)


def grid_to_blueprint(grid: np.ndarray, vocab: Vocab,
                      label: str = "predicted patch (factorio-patch-inpaint)",
                      version: int = VERSION_1_1) -> dict:
    """Convert a token grid back into a minimal Factorio blueprint dict.

    Anchors only (no multi-tile collision handling or validity checks), but the
    structure is a real importable blueprint. ``version`` MUST match the grid's
    direction encoding: a 2.0-native grid (16-direction names/dirs) needs
    ``VERSION_2_0`` so the game/FBSR doesn't re-migrate (and corrupt) directions.
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
    return {"blueprint": {"item": "blueprint", "entities": entities,
                          "version": version, "label": label}}


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
                for tag, g in (("prediction", pred_full), ("target", y)):
                    bp = grid_to_blueprint(g, vocab, label=f"{args.split}_{i:03d} {tag}")
                    stem = grids_dir / f"{args.split}_{i:03d}_{tag}"
                    bp_str = encode_blueprint_string(bp)
                    stem.with_suffix(".blueprint.txt").write_text(bp_str)   # importable into Factorio
                    stem.with_suffix(".blueprint.json").write_text(json.dumps(bp))

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
