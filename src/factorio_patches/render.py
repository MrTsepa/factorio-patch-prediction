"""Render token grids to PNGs for visual inspection.

Not meant to look like Factorio — just enough to understand model behavior.
EMPTY = white, MASK = gray, UNK = magenta, entities colored by family with
small direction arrows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .vocab import EMPTY, EMPTY_ID, MASK, MASK_ID, UNK, UNK_ID, Vocab, split_token

# Exact-name colors for the belt families (the first-scope focus): yellow/red/blue tiers.
NAME_COLORS = {
    "transport-belt": (240, 205, 50), "fast-transport-belt": (225, 90, 60),
    "express-transport-belt": (70, 140, 225),
    "underground-belt": (190, 160, 40), "fast-underground-belt": (170, 65, 45),
    "express-underground-belt": (45, 100, 175),
    "splitter": (250, 225, 120), "fast-splitter": (240, 140, 120),
    "express-splitter": (130, 180, 240),
}
# Family substring -> color (checked in order; first match wins).
FAMILY_COLORS = {
    "underground-belt": (190, 160, 40), "transport-belt": (240, 205, 50),
    "splitter": (250, 225, 120),
    "inserter": (90, 160, 220), "electric-pole": (150, 220, 240), "substation": (110, 190, 215),
    "assembling-machine": (80, 190, 110), "chemical-plant": (120, 200, 150),
    "oil-refinery": (100, 170, 130), "furnace": (220, 120, 70),
    "pipe": (80, 200, 190), "pump": (70, 175, 170),
    "rail": (150, 130, 120), "train-stop": (170, 110, 90),
    "locomotive": (120, 100, 95), "wagon": (135, 115, 110),
    "beacon": (200, 150, 220), "roboport": (180, 140, 90),
    "chest": (200, 180, 130), "lab": (160, 130, 200), "wall": (140, 140, 140),
    "turret": (110, 90, 90), "reactor": (230, 90, 90), "accumulator": (120, 150, 200),
    "solar-panel": (60, 70, 90), "boiler": (200, 100, 80), "generator": (210, 170, 90),
    "radar": (140, 160, 110), "lamp": (235, 225, 160),
}
DIR8 = {0: (0, -1), 1: (1, -1), 2: (1, 0), 3: (1, 1),
        4: (0, 1), 5: (-1, 1), 6: (-1, 0), 7: (-1, -1)}


def _hash_color(name: str):
    h = abs(hash(name))
    # Pastel-ish, avoid near-white / near-black.
    return (60 + h % 150, 60 + (h // 150) % 150, 60 + (h // 22500) % 150)


def token_color(token: str):
    if token == EMPTY:
        return (255, 255, 255)
    if token == MASK:
        return (140, 140, 140)
    if token == UNK:
        return (230, 60, 200)
    name, _ = split_token(token)
    if name in NAME_COLORS:
        return NAME_COLORS[name]
    for fam, col in FAMILY_COLORS.items():
        if fam in name:
            return col
    return _hash_color(name)


def _direction_of(token: str):
    _, d = split_token(token)
    return d


def render_grid(grid: np.ndarray, vocab: Vocab, cell: int = 14, draw_dirs: bool = True,
                mask: np.ndarray | None = None, grid_lines: bool = True) -> Image.Image:
    """Render an int token grid to a PIL image (cell px per cell)."""
    H, W = grid.shape
    img = Image.new("RGB", (W * cell, H * cell), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            tid = int(grid[r, c])
            tok = vocab.decode(tid)
            col = token_color(tok)
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell - 1, y0 + cell - 1
            d.rectangle([x0, y0, x1, y1], fill=col)
            if draw_dirs and tid not in (EMPTY_ID, MASK_ID) and cell >= 8:
                dir_ = _direction_of(tok)
                if dir_ is not None:
                    vx, vy = DIR8.get(int(dir_) % 8, (0, 0))
                    if vx or vy:
                        cx, cy = x0 + cell / 2, y0 + cell / 2
                        L = cell * 0.34
                        ex, ey = cx + vx * L, cy + vy * L
                        d.line([cx - vx * L * 0.4, cy - vy * L * 0.4, ex, ey],
                               fill=(20, 20, 20), width=max(1, cell // 10))
    if grid_lines and cell >= 8:
        for r in range(H + 1):
            d.line([0, r * cell, W * cell, r * cell], fill=(225, 225, 225))
        for c in range(W + 1):
            d.line([c * cell, 0, c * cell, H * cell], fill=(225, 225, 225))
    if mask is not None:
        ys, xs = np.where(mask)
        if len(ys):
            r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
            d.rectangle([c0 * cell, r0 * cell, (c1 + 1) * cell - 1, (r1 + 1) * cell - 1],
                        outline=(220, 30, 30), width=max(2, cell // 6))
    return img


def render_diff(y: np.ndarray, pred: np.ndarray, mask: np.ndarray, cell: int = 14) -> Image.Image:
    """Green = correct inside mask, red = wrong inside mask, faint elsewhere."""
    H, W = y.shape
    img = Image.new("RGB", (W * cell, H * cell), (245, 245, 245))
    d = ImageDraw.Draw(img)
    for r in range(H):
        for c in range(W):
            x0, y0 = c * cell, r * cell
            x1, y1 = x0 + cell - 1, y0 + cell - 1
            if mask[r, c]:
                correct = (y[r, c] == pred[r, c])
                both_empty = (y[r, c] == EMPTY_ID and pred[r, c] == EMPTY_ID)
                if both_empty:
                    col = (210, 235, 210)
                elif correct:
                    col = (40, 170, 70)
                else:
                    col = (210, 60, 50)
            else:
                col = (245, 245, 245) if y[r, c] == EMPTY_ID else (220, 220, 220)
            d.rectangle([x0, y0, x1, y1], fill=col)
    return img


def hconcat(images, pad: int = 8, bg=(255, 255, 255)) -> Image.Image:
    h = max(im.height for im in images)
    w = sum(im.width for im in images) + pad * (len(images) - 1)
    out = Image.new("RGB", (w, h), bg)
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + pad
    return out


def render_prediction_set(x, y, pred, mask, vocab: Vocab, out_dir: Path, name: str,
                          cell: int = 14, combined: bool = True):
    """Write input/target/prediction/diff PNGs (and an optional combined panel)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x, y, pred = np.asarray(x), np.asarray(y), np.asarray(pred)
    mask = np.asarray(mask).astype(bool)
    img_in = render_grid(x, vocab, cell=cell, mask=mask)
    img_tg = render_grid(y, vocab, cell=cell, mask=mask)
    img_pr = render_grid(pred, vocab, cell=cell, mask=mask)
    img_df = render_diff(y, pred, mask, cell=cell)
    img_in.save(out_dir / f"{name}_input.png")
    img_tg.save(out_dir / f"{name}_target.png")
    img_pr.save(out_dir / f"{name}_prediction.png")
    img_df.save(out_dir / f"{name}_diff.png")
    if combined:
        hconcat([img_in, img_tg, img_pr, img_df]).save(out_dir / f"{name}_panel.png")


def main(argv=None) -> int:
    from .dataset import load_dataset, make_datasets
    ap = argparse.ArgumentParser(description="Render dataset samples (input/target/mask) to PNGs.")
    ap.add_argument("--dataset", type=Path, required=True, help="data/processed/dataset.pt")
    ap.add_argument("--out", type=Path, default=Path("outputs/samples"))
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--cell", type=int, default=14)
    args = ap.parse_args(argv)

    payload = load_dataset(args.dataset)
    ds = make_datasets(payload)
    vocab = ds["vocab"]
    split = ds[args.split]
    if split is None:
        print(f"split '{args.split}' is empty")
        return 1
    args.out.mkdir(parents=True, exist_ok=True)
    n = min(args.num, len(split))
    for i in range(n):
        s = split[i]
        x, y, m = s["x"].numpy(), s["y"].numpy(), s["mask"].numpy()
        img = hconcat([
            render_grid(x, vocab, cell=args.cell, mask=m),
            render_grid(y, vocab, cell=args.cell, mask=m),
        ])
        img.save(args.out / f"sample_{i:03d}.png")
    print(f"Wrote {n} sample panel(s) (input | target) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
