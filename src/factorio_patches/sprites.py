"""Optional sprite-based rendering using real Factorio entity icons.

This renders grids with the actual base-game icons instead of colored cells, for
nicer visualizations. The icons are NOT shipped with this repo (they are Wube's
copyrighted assets) — point ``--sprites`` at a local Factorio icon directory, e.g.
``factorio.app/Contents/data/base/graphics/icons`` or a mod's extracted icons.

Icons are 64x64 with a horizontal mipmap strip (e.g. 120x64); we crop the leading
square. Two refinements over a flat one-icon-per-cell render:

* **footprint** — multi-tile entities (assemblers, beacons, furnaces, ...) are
  scaled to their tile size and drawn centered on their anchor;
* **rotation** — directional entities (belts, inserters, splitters, ...) are
  rotated by their ``direction``.

Large entities are drawn first so small ones (belts/inserters) land on top.
Missing entities fall back to the abstract colored-cell renderer.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .render import token_color
from .vocab import EMPTY_ID, MASK_ID, UNK_ID, split_token

# Factorio 1.1 (our blueprints) -> 2.0 base icon name renames.
ALIASES = {
    "stack-inserter": "bulk-inserter",
    "stack-filter-inserter": "bulk-inserter",
    "filter-inserter": "fast-inserter",
    "straight-rail": "rail",
    "curved-rail": "rail",
    "stone-wall": "wall",
    "logistic-chest-storage": "storage-chest",
    "logistic-chest-requester": "requester-chest",
    "logistic-chest-buffer": "buffer-chest",
    "logistic-chest-active-provider": "active-provider-chest",
    "logistic-chest-passive-provider": "passive-provider-chest",
    "heat-exchanger": "heat-boiler",
}

# Tile footprint (width_x, height_y) in NORTH orientation. Default is 1x1.
SIZES = {
    "accumulator": (2, 2), "big-electric-pole": (2, 2), "substation": (2, 2),
    "stone-furnace": (2, 2), "steel-furnace": (2, 2), "gun-turret": (2, 2),
    "laser-turret": (2, 2), "straight-rail": (2, 2), "curved-rail": (2, 2),
    "train-stop": (2, 2),
    "assembling-machine-1": (3, 3), "assembling-machine-2": (3, 3),
    "assembling-machine-3": (3, 3), "beacon": (3, 3), "electric-furnace": (3, 3),
    "chemical-plant": (3, 3), "centrifuge": (3, 3), "lab": (3, 3),
    "radar": (3, 3), "solar-panel": (3, 3), "storage-tank": (3, 3),
    "electric-mining-drill": (3, 3),
    "roboport": (4, 4), "oil-refinery": (5, 5),
    "splitter": (2, 1), "fast-splitter": (2, 1), "express-splitter": (2, 1),
    "boiler": (3, 2), "heat-exchanger": (3, 2), "steam-turbine": (3, 5),
    "pump": (1, 2), "arithmetic-combinator": (1, 2), "decider-combinator": (1, 2),
    "fluid-wagon": (2, 6), "cargo-wagon": (2, 6), "locomotive": (2, 6),
}
# Entities whose icon should rotate with `direction` (others render upright).
ROTATABLE = ("belt", "inserter", "splitter", "pump", "pipe-to-ground", "gate", "wall")

DEFAULT_ICONS = "/Users/mrtsepa/Workspace/factorio.app/Contents/data/base/graphics/icons"


def tile_size(name: str) -> tuple[int, int]:
    return SIZES.get(name, (1, 1))


class SpriteRenderer:
    def __init__(self, icons_dir: str | None = None, bg=(236, 238, 240)):
        self.dir = Path(icons_dir or os.environ.get("FACTORIO_ICONS", DEFAULT_ICONS))
        self.bg = bg
        self._resolved: dict[str, str | None] = {}
        self._base: dict[str, Image.Image | None] = {}
        self._cache: dict[tuple, Image.Image | None] = {}
        self.available = self.dir.is_dir()

    def _resolve(self, name: str):
        if name in self._resolved:
            return self._resolved[name]
        cand = name if (self.dir / f"{name}.png").exists() else None
        if cand is None:
            a = ALIASES.get(name)
            if a and (self.dir / f"{a}.png").exists():
                cand = a
        self._resolved[name] = cand
        return cand

    def _base_icon(self, name: str):
        if name in self._base:
            return self._base[name]
        fn = self._resolve(name)
        out = None
        if fn:
            try:
                im = Image.open(self.dir / f"{fn}.png").convert("RGBA")
                s = min(im.width, im.height)        # crop leading square (full-res mip)
                out = im.crop((0, 0, s, s))
            except Exception:
                out = None
        self._base[name] = out
        return out

    def sprite(self, name: str, wpx: int, hpx: int, angle: int):
        """Icon scaled to (wpx,hpx) and rotated `angle` degrees clockwise."""
        key = (name, wpx, hpx, angle)
        if key in self._cache:
            return self._cache[key]
        base = self._base_icon(name)
        out = None
        if base is not None:
            im = base.resize((max(1, wpx), max(1, hpx)), Image.LANCZOS)
            if angle:
                im = im.rotate(-angle, expand=True, resample=Image.BICUBIC)
            out = im
        self._cache[key] = out
        return out

    def coverage(self, vocab) -> float:
        names = {split_token(t)[0] for t in vocab.itos if "|" in t}
        return (sum(1 for n in names if self._resolve(n)) / len(names)) if names else 0.0

    def render_grid(self, grid, vocab, cell: int = 24, mask=None, draw_dirs: bool = True):
        H, W = grid.shape
        img = Image.new("RGB", (W * cell, H * cell), self.bg)
        d = ImageDraw.Draw(img)
        for r in range(H + 1):
            d.line([0, r * cell, W * cell, r * cell], fill=(248, 249, 250))
        for c in range(W + 1):
            d.line([c * cell, 0, c * cell, H * cell], fill=(248, 249, 250))

        # Collect entities, draw largest footprint first so belts/inserters land on top.
        items = []
        for r in range(H):
            for c in range(W):
                tid = int(grid[r, c])
                if tid in (EMPTY_ID, MASK_ID):
                    continue
                name, dir_ = split_token(vocab.decode(tid))
                w, h = tile_size(name)
                items.append((w * h, r, c, tid, name, dir_, w, h))
        items.sort(key=lambda t: -t[0])

        for _, r, c, tid, name, dir_, w, h in items:
            cx, cy = c * cell + cell / 2, r * cell + cell / 2
            angle = 0
            if draw_dirs and dir_ and any(k in name for k in ROTATABLE):
                angle = (int(dir_) * 45) % 360
            sp = None if tid == UNK_ID else self.sprite(name, w * cell, h * cell, angle)
            if sp is not None:
                img.paste(sp, (int(cx - sp.width / 2), int(cy - sp.height / 2)), sp)
            else:
                col = token_color(vocab.decode(tid))
                d.rectangle([cx - w * cell / 2 + 1, cy - h * cell / 2 + 1,
                             cx + w * cell / 2 - 2, cy + h * cell / 2 - 2], fill=col)

        # MASK on top so the hole reads cleanly even if a neighbour's footprint bleeds in.
        for r in range(H):
            for c in range(W):
                if int(grid[r, c]) == MASK_ID:
                    d.rectangle([c * cell, r * cell, c * cell + cell - 1, r * cell + cell - 1],
                                fill=(150, 150, 150))
        if mask is not None:
            ys, xs = np.where(mask)
            if len(ys):
                r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
                d.rectangle([c0 * cell, r0 * cell, (c1 + 1) * cell - 1, (r1 + 1) * cell - 1],
                            outline=(220, 30, 30), width=max(2, cell // 8))
        return img
