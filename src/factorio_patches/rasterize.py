"""Rasterize a blueprint into a 2D integer token grid.

v0 uses entity *anchors* only (one cell per entity, no multi-tile collision
boxes). Positions are rounded to the nearest integer cell and normalized so the
min x/y sits at 0. Cell collisions keep the first entity and are counted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .vocab import EMPTY_ID, UNK, Vocab, entity_token, io_for, split_token


class RasterizeTooLarge(ValueError):
    """Blueprint bounding box exceeds the allowed grid size."""


@dataclass
class RasterizedBlueprint:
    grid: np.ndarray            # int16 [H, W], token ids (EMPTY_ID filled)
    height: int
    width: int
    bbox: tuple                 # (min_x, min_y, max_x, max_y) of rounded cells
    n_entities: int
    n_cells_filled: int
    n_collisions: int
    n_unk: int
    label: str | None = None
    source_hash: str | None = None
    bp_id: str | None = None
    group_id: str | None = None   # anti-leakage split group (connected component)
    source: str | None = None     # corpus label (factoriobin / factorioprints)
    meta: dict = field(default_factory=dict)

    @property
    def density(self) -> float:
        area = self.height * self.width
        return (self.n_cells_filled / area) if area else 0.0


def _round_cell(v: float) -> int:
    # Standard round-half-up (avoids numpy banker's rounding), works for negatives.
    return int(np.floor(v + 0.5))


def rasterize_blueprint(bp: dict, vocab: Vocab, max_dim: int = 512) -> RasterizedBlueprint:
    """Rasterize one extracted-blueprint record into a token grid."""
    entities = bp.get("entities") or []
    cols, rows, tokens = [], [], []
    for e in entities:
        pos = e.get("position") or {}
        x = pos.get("x")
        y = pos.get("y")
        if x is None or y is None:
            continue
        nm = e.get("name")
        cols.append(_round_cell(x))
        rows.append(_round_cell(y))
        tokens.append(entity_token(nm, e.get("direction"), io_for(nm, e)))

    if not cols:
        grid = np.full((1, 1), EMPTY_ID, dtype=np.int16)
        return RasterizedBlueprint(grid, 1, 1, (0, 0, 0, 0), 0, 0, 0, 0,
                                   bp.get("label"), bp.get("source_hash"), bp.get("id"),
                                   bp.get("group_id"), bp.get("source"))

    cols = np.asarray(cols, dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    min_x, min_y = int(cols.min()), int(rows.min())
    max_x, max_y = int(cols.max()), int(rows.max())
    W = max_x - min_x + 1
    H = max_y - min_y + 1
    if H > max_dim or W > max_dim:
        raise RasterizeTooLarge(f"grid {H}x{W} exceeds max_dim={max_dim}")

    cols -= min_x
    rows -= min_y
    grid = np.full((H, W), EMPTY_ID, dtype=np.int16)
    filled = 0
    collisions = 0
    n_unk = 0
    for r, c, tok in zip(rows.tolist(), cols.tolist(), tokens):
        if grid[r, c] != EMPTY_ID:
            collisions += 1
            continue
        tid = vocab.encode(tok)
        grid[r, c] = tid
        filled += 1
        if tid == vocab.unk_id:
            n_unk += 1

    return RasterizedBlueprint(
        grid=grid, height=H, width=W, bbox=(min_x, min_y, max_x, max_y),
        n_entities=len(tokens), n_cells_filled=filled, n_collisions=collisions, n_unk=n_unk,
        label=bp.get("label"), source_hash=bp.get("source_hash"), bp_id=bp.get("id"),
        group_id=bp.get("group_id"), source=bp.get("source"),
    )


def iter_rasterized(blueprints_jsonl: Path, vocab: Vocab, max_dim: int = 512,
                    min_entities: int = 1):
    """Yield RasterizedBlueprint for each record, skipping oversized ones."""
    n_skipped = 0
    with open(blueprints_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bp = json.loads(line)
            if bp.get("n_entities", len(bp.get("entities", []))) < min_entities:
                continue
            try:
                rb = rasterize_blueprint(bp, vocab, max_dim=max_dim)
            except RasterizeTooLarge:
                n_skipped += 1
                continue
            yield rb
    if n_skipped:
        print(f"  [note] skipped {n_skipped} blueprint(s) larger than max_dim={max_dim}")


def compute_stats(blueprints_jsonl: Path, vocab: Vocab, max_dim: int = 512) -> dict:
    n = 0
    ents, hs, ws, dens = [], [], [], []
    tot_cells = 0
    tot_coll = 0
    tot_unk = 0
    tok_counter: Counter = Counter()
    for rb in iter_rasterized(blueprints_jsonl, vocab, max_dim=max_dim):
        n += 1
        ents.append(rb.n_entities)
        hs.append(rb.height)
        ws.append(rb.width)
        dens.append(rb.density)
        tot_cells += rb.n_cells_filled
        tot_coll += rb.n_collisions
        tot_unk += rb.n_unk
        ids, counts = np.unique(rb.grid, return_counts=True)
        for i, c in zip(ids.tolist(), counts.tolist()):
            if i != EMPTY_ID:
                tok_counter[vocab.decode(i)] += c

    def pctl(a, q):
        return float(np.percentile(a, q)) if a else 0.0

    placed = tot_cells + tot_coll
    stats = {
        "n_blueprints": n,
        "entities": {"min": min(ents) if ents else 0, "median": pctl(ents, 50),
                     "p90": pctl(ents, 90), "max": max(ents) if ents else 0},
        "grid_height": {"min": min(hs) if hs else 0, "median": pctl(hs, 50),
                        "p90": pctl(hs, 90), "max": max(hs) if hs else 0},
        "grid_width": {"min": min(ws) if ws else 0, "median": pctl(ws, 50),
                       "p90": pctl(ws, 90), "max": max(ws) if ws else 0},
        "mean_density": float(np.mean(dens)) if dens else 0.0,
        "cells_filled": tot_cells,
        "collisions": tot_coll,
        "collision_rate": (tot_coll / placed) if placed else 0.0,
        "unk_cells": tot_unk,
        "unk_rate": (tot_unk / tot_cells) if tot_cells else 0.0,
        "vocab_size": len(vocab),
        "top_tokens": tok_counter.most_common(20),
    }
    return stats


def print_stats(stats: dict) -> None:
    print(f"Rasterization stats:")
    print(f"  blueprints        : {stats['n_blueprints']}")
    e = stats["entities"]
    print(f"  entities/bp       : min={e['min']} median={e['median']:.0f} p90={e['p90']:.0f} max={e['max']}")
    h, w = stats["grid_height"], stats["grid_width"]
    print(f"  grid H            : min={h['min']} median={h['median']:.0f} p90={h['p90']:.0f} max={h['max']}")
    print(f"  grid W            : min={w['min']} median={w['median']:.0f} p90={w['p90']:.0f} max={w['max']}")
    print(f"  mean density      : {stats['mean_density']:.3f}")
    print(f"  cells filled      : {stats['cells_filled']}")
    print(f"  collision rate    : {stats['collision_rate']:.3%} ({stats['collisions']} collisions)")
    print(f"  UNK rate          : {stats['unk_rate']:.3%} ({stats['unk_cells']} unk cells)")
    print(f"  vocab size        : {stats['vocab_size']}")
    print("  most common tokens:")
    for tok, c in stats["top_tokens"]:
        print(f"    {tok:32s} {c:>7d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rasterize blueprints and print grid stats.")
    ap.add_argument("--blueprints", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--max-dim", type=int, default=512)
    args = ap.parse_args(argv)
    vocab = Vocab.load(args.vocab)
    stats = compute_stats(args.blueprints, vocab, max_dim=args.max_dim)
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
