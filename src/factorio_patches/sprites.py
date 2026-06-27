"""Optional sprite-based rendering using real Factorio entity icons.

This renders grids with the actual base-game icons instead of colored cells, for
nicer visualizations. The icons are NOT shipped with this repo (they are Wube's
copyrighted assets) — point ``--sprites`` at a local Factorio icon directory, e.g.
``factorio.app/Contents/data/base/graphics/icons`` or a mod's extracted icons.

Icons are 64x64 with a horizontal mipmap strip (e.g. 120x64); we crop the leading
square. Missing entities fall back to the abstract colored-cell renderer, so this
degrades gracefully if an icon (or the whole directory) is absent.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image, ImageDraw

from .render import DIR8, token_color
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
DIRECTIONAL = ("belt", "inserter", "splitter", "pump", "pipe-to-ground")

DEFAULT_ICONS = "/Users/mrtsepa/Workspace/factorio.app/Contents/data/base/graphics/icons"


class SpriteRenderer:
    def __init__(self, icons_dir: str | None = None, bg=(236, 238, 240)):
        self.dir = Path(icons_dir or os.environ.get("FACTORIO_ICONS", DEFAULT_ICONS))
        self.bg = bg
        self._resolved: dict[str, str | None] = {}
        self._cache: dict[tuple[str, int], Image.Image | None] = {}
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

    def icon(self, name: str, size: int):
        key = (name, size)
        if key in self._cache:
            return self._cache[key]
        fn = self._resolve(name)
        out = None
        if fn:
            try:
                im = Image.open(self.dir / f"{fn}.png").convert("RGBA")
                s = min(im.width, im.height)        # crop leading square (full-res mip)
                im = im.crop((0, 0, s, s)).resize((size, size), Image.LANCZOS)
                out = im
            except Exception:
                out = None
        self._cache[key] = out
        return out

    def coverage(self, vocab) -> float:
        names = {split_token(t)[0] for t in vocab.itos if "|" in t}
        if not names:
            return 0.0
        return sum(1 for n in names if self._resolve(n)) / len(names)

    def render_grid(self, grid, vocab, cell: int = 24, mask=None, draw_dirs: bool = True):
        H, W = grid.shape
        img = Image.new("RGB", (W * cell, H * cell), self.bg)
        d = ImageDraw.Draw(img)
        # faint grid
        for r in range(H + 1):
            d.line([0, r * cell, W * cell, r * cell], fill=(248, 249, 250))
        for c in range(W + 1):
            d.line([c * cell, 0, c * cell, H * cell], fill=(248, 249, 250))
        for r in range(H):
            for c in range(W):
                tid = int(grid[r, c])
                x0, y0 = c * cell, r * cell
                if tid == EMPTY_ID:
                    continue
                if tid == MASK_ID:
                    d.rectangle([x0, y0, x0 + cell - 1, y0 + cell - 1], fill=(150, 150, 150))
                    continue
                tok = vocab.decode(tid)
                name, dir_ = split_token(tok)
                ic = None if tid == UNK_ID else self.icon(name, cell)
                if ic is not None:
                    img.paste(ic, (x0, y0), ic)
                else:
                    d.rectangle([x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2],
                                fill=token_color(tok))
                if draw_dirs and dir_ and cell >= 14 and any(k in name for k in DIRECTIONAL):
                    vx, vy = DIR8.get(int(dir_) % 8, (0, 0))
                    if vx or vy:
                        cx, cy = x0 + cell / 2, y0 + cell / 2
                        L = cell * 0.42
                        d.line([cx - vx * L * 0.5, cy - vy * L * 0.5, cx + vx * L, cy + vy * L],
                               fill=(15, 15, 15), width=max(1, cell // 12))
        if mask is not None:
            import numpy as np
            ys, xs = np.where(mask)
            if len(ys):
                r0, r1, c0, c1 = ys.min(), ys.max(), xs.min(), xs.max()
                d.rectangle([c0 * cell, r0 * cell, (c1 + 1) * cell - 1, (r1 + 1) * cell - 1],
                            outline=(220, 30, 30), width=max(2, cell // 8))
        return img
