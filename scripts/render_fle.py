#!/usr/bin/env python
"""Render a blueprint string to a GAME-ACCURATE PNG using the factorio-learning-
environment (FLE) sprite renderer.

This is how blueprints are "normally" rendered (real entity sprites, footprints,
rotations, shadows) — the same idea behind FactorioBin / the Reddit BlueprintBot
(which use the Java FBSR engine); here we use FLE's pure-Python renderer.

IMPORTANT — this script imports `fle`, so it must run with FLE's interpreter,
NOT this project's venv. One-time setup (see docs/findings.md / DEMO.md):

  1. FLE's render-time sprite set ships incomplete (icons only). Generate the
     entity bodies from FLE's local `.fle/spritemaps` (no internet / no game):
       # relink the pre-transcoded sprite cache to YOUR machine's path so the
       # missing `basisu` transcoder is never needed, then:
       FLE/.venv/bin/python -c "from fle.agents.data.sprites.download import \
         generate_sprites; generate_sprites(input_dir='ABS/.fle/spritemaps', \
         output_dir='ABS/.fle/sprites')"
  2. Run this script with FLE's python and FLE_SPRITES_DIR pointing at .fle/sprites:
       FLE_SPRITES_DIR=ABS/.fle/sprites  FLE/.venv/bin/python scripts/render_fle.py \
         pred.blueprint.txt out.png [--scale 32]

Our blueprints are Factorio 1.1 (8-direction). FLE's per-entity renderers hardcode
2.0 direction tables and have a couple of dict-vs-object bugs; this script patches
them at runtime (no edits to the FLE repo).
"""

from __future__ import annotations

import argparse
import importlib
import os
import pkgutil
import sys
from pathlib import Path


def _patch_fle():
    """Make FLE's renderers tolerant of Factorio 1.1 (8-direction) blueprints."""
    import fle.env.tools.admin.render.renderers as renderers_pkg

    nesw = {0: "north", 1: "north", 2: "east", 3: "east", 4: "south", 5: "south",
            6: "west", 7: "west", 8: "south", 10: "west", 12: "west", 14: "north"}
    urdl = {0: "up", 1: "up", 2: "right", 3: "right", 4: "down", 5: "down",
            6: "left", 7: "left", 8: "down", 10: "left", 12: "left", 14: "up"}
    for m in pkgutil.iter_modules(renderers_pkg.__path__):
        mod = importlib.import_module(f"{renderers_pkg.__name__}.{m.name}")
        for attr in ("DIRECTIONS", "RELATIVE_DIRECTIONS"):
            tbl = getattr(mod, attr, None)
            if isinstance(tbl, dict):
                setattr(mod, attr, dict(urdl if "up" in tbl.values() else nesw))


# Rail-family / rolling stock: FLE's _render_rails assumes Entity objects and
# crashes on dict entities; drop them (not the focus of this POC anyway).
SKIP = {"straight-rail", "curved-rail", "rail", "rail-signal", "rail-chain-signal",
        "train-stop", "locomotive", "cargo-wagon", "fluid-wagon", "artillery-wagon"}


def entities_of(data: dict) -> list:
    if "blueprint" in data:
        ents = data["blueprint"].get("entities", [])
    else:
        ents = []
        for b in data.get("blueprint_book", {}).get("blueprints", []):
            if "blueprint" in b:
                ents += b["blueprint"].get("entities", [])
    out = []
    for e in ents:
        if e.get("name") in SKIP:
            continue
        # FLE's belt renderer reads underground-belt['type'] for neighbours.
        if "underground-belt" in e.get("name", "") and "type" not in e:
            e = {**e, "type": "input"}
        out.append(e)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Render a blueprint string/file to a game-accurate PNG via FLE.")
    ap.add_argument("source", help="blueprint string, or a path to a .txt containing one")
    ap.add_argument("out", help="output PNG path")
    ap.add_argument("--scale", type=int, default=32, help="pixels per tile")
    ap.add_argument("--sprites", default=os.environ.get("FLE_SPRITES_DIR"),
                    help="FLE .fle/sprites dir (or set FLE_SPRITES_DIR)")
    args = ap.parse_args(argv)
    if not args.sprites:
        ap.error("set --sprites or FLE_SPRITES_DIR to FLE's .fle/sprites directory")
    os.environ.setdefault("FLE_SPRITES_DIR", args.sprites)

    _patch_fle()
    from fle.env.tools.admin.render.utils import parse_blueprint
    from fle.env.tools.admin.render.renderer import Renderer
    from fle.env.tools.admin.render.image_resolver import ImageResolver

    s = Path(args.source).read_text().strip() if Path(args.source).exists() else args.source
    ents = entities_of(parse_blueprint(s))
    if not ents:
        print("no renderable entities", file=sys.stderr)
        return 1
    r = Renderer(entities=ents, sprites_dir=Path(args.sprites))
    sz = r.get_size()
    img = r.render(int((sz["width"] + 2) * args.scale), int((sz["height"] + 2) * args.scale),
                   ImageResolver(args.sprites))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"saved {args.out} {img.size} ({len(ents)} entities)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
