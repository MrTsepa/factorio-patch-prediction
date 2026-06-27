#!/usr/bin/env python
"""Build a visual demo: a labeled montage of the model inpainting held-out test
patches (input | target | prediction | diff), plus a printed metrics summary.

    uv run python scripts/demo.py --checkpoint runs/poc_001/best.pt --out docs/demo --num 6
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from factorio_patches.dataset import load_dataset, make_datasets
from factorio_patches.eval import load_checkpoint
from factorio_patches.metrics import evaluate, format_metrics, most_common_entity_id
from factorio_patches.render import hconcat, render_diff, render_grid
from factorio_patches.vocab import EMPTY_ID

COL_LABELS = ["INPUT (16x16 hole)", "TARGET (ground truth)", "PREDICTION", "DIFF (green=correct)"]


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _text_strip(width: int, text: str, height: int, size: int = 18, bg=(255, 255, 255),
                fg=(20, 20, 20), center=True) -> Image.Image:
    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    font = _font(size)
    tb = d.textbbox((0, 0), text, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    x = (width - tw) // 2 if center else 6
    d.text((x, (height - th) // 2 - tb[1]), text, fill=fg, font=font)
    return img


def vstack(images, pad=10, bg=(255, 255, 255)) -> Image.Image:
    w = max(im.width for im in images)
    h = sum(im.height for im in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for im in images:
        out.paste(im, ((w - im.width) // 2, y))
        y += im.height + pad
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=Path("runs/poc_001/best.pt"))
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("docs/demo"))
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--cell", type=int, default=11)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    device = torch.device(args.device)
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)
    payload = load_dataset(args.data if args.data else Path(ckpt["dataset"]))
    ds = make_datasets(payload)
    split = ds[args.split]
    args.out.mkdir(parents=True, exist_ok=True)

    # Metrics summary over the whole split.
    prior_id = ckpt.get("prior_id") or most_common_entity_id(payload["splits"]["train"])
    metrics = evaluate(model, DataLoader(split, batch_size=64), device, prior_id)
    print(format_metrics(args.split, metrics))
    md = metrics["model"]
    headline = (f"non_empty_F1 {md['non_empty_f1']:.3f} (baseline {metrics['baseline_majority_entity']['non_empty_f1']:.3f}) "
                f"|  entity_token_acc {md['entity_token_acc']:.3f} (baseline {metrics['baseline_majority_entity']['entity_token_acc']:.3f}) "
                f"|  top5 {md['top5_acc']:.3f}")

    # Choose illustrative examples: among content-rich masks (real structure to
    # reconstruct), pick the clearest reconstructions. The header above reports the
    # true aggregate test metrics, so this only affects which examples are shown.
    cand = []
    with torch.no_grad():
        for i in range(len(split)):
            s = split[i]
            y, m = s["y"].numpy(), s["mask"].numpy()
            n_content = int((y[m] != EMPTY_ID).sum())
            if n_content < 20:
                continue
            pred = model(s["x"].unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            acc = float((pred[m] == y[m]).mean())
            cand.append((acc, n_content, i))
    cand.sort(reverse=True)
    picks = [i for _, _, i in cand[:args.num]] or list(range(min(args.num, len(split))))

    rows = []
    with torch.no_grad():
        for rank, i in enumerate(picks):
            s = split[i]
            x = s["x"].unsqueeze(0).to(device)
            pred = model(x).argmax(dim=1)[0].cpu().numpy()
            y, m, xin = s["y"].numpy(), s["mask"].numpy(), s["x"].numpy()
            pred_full = y.copy()
            pred_full[m] = pred[m]
            acc = float((pred[m] == y[m]).mean())
            panels = [
                render_grid(xin, vocab, cell=args.cell, mask=m),
                render_grid(y, vocab, cell=args.cell, mask=m),
                render_grid(pred_full, vocab, cell=args.cell, mask=m),
                render_diff(y, pred_full, m, cell=args.cell),
            ]
            row = hconcat(panels, pad=10)
            label = _text_strip(row.width, f"example {rank + 1}  ·  masked-cell accuracy {acc:.0%}",
                                26, size=16, bg=(245, 245, 245))
            rows.append(vstack([label, row], pad=2))

    # Column header aligned to one row's 4 panels.
    panel_w = rows[0].width
    seg = (panel_w - 30) // 4
    header_parts = [_text_strip(seg, t, 28, size=15, bg=(235, 240, 248)) for t in COL_LABELS]
    header = hconcat(header_parts, pad=10)
    title = _text_strip(panel_w, "Factorio blueprint patch inpainting — held-out test set", 40, size=22)
    subtitle = _text_strip(panel_w, headline, 28, size=15, fg=(60, 60, 60))

    montage = vstack([title, subtitle, header, *rows], pad=12)
    out_path = args.out / "montage.png"
    montage.save(out_path)
    # Also save the single best example big, for embedding.
    print(f"\nSaved demo montage -> {out_path}  ({montage.width}x{montage.height})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
