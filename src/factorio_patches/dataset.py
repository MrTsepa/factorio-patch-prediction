"""Masked-patch dataset for blueprint inpainting.

Each example is a ``crop_size`` x ``crop_size`` window taken from a blueprint
grid, with a ``mask_size`` x ``mask_size`` rectangle replaced by ``MASK`` in the
input. The target is the original window; loss is computed only inside the mask.

The split is done at the *blueprint* level (not the patch level) to avoid train
/ val / test leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .rasterize import iter_rasterized
from .vocab import EMPTY_ID, MASK_ID, Vocab


# --------------------------------------------------------------------------- #
# Building / saving the processed grids + split
# --------------------------------------------------------------------------- #
def build_dataset(
    blueprints_jsonl: Path,
    vocab: Vocab,
    out_path: Path,
    crop_size: int = 64,
    mask_size: int = 16,
    max_dim: int = 512,
    min_dim: int = 0,
    min_entities: int = 4,
    min_filled: int = 4,
    min_invocab_frac: float = 0.0,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
) -> dict:
    """Rasterize, filter, split by blueprint, and save to ``out_path`` (.pt).

    ``min_invocab_frac`` drops blueprints whose occupied cells are mostly UNK
    (e.g. rail yards under a logistics vocab), keeping the inpainting task on
    in-vocabulary structure.
    """
    grids, metas = [], []
    n_unk_filtered = 0
    n_small_filtered = 0
    for rb in iter_rasterized(blueprints_jsonl, vocab, max_dim=max_dim, min_entities=min_entities):
        if rb.n_cells_filled < min_filled:
            continue
        if min(rb.height, rb.width) < min_dim:
            n_small_filtered += 1
            continue
        invocab_frac = (rb.n_cells_filled - rb.n_unk) / rb.n_cells_filled if rb.n_cells_filled else 0.0
        if invocab_frac < min_invocab_frac:
            n_unk_filtered += 1
            continue
        grids.append(rb.grid.astype(np.int16))
        metas.append({
            "id": rb.bp_id, "label": rb.label, "source_hash": rb.source_hash,
            "group_id": rb.group_id, "source": rb.source,
            "height": rb.height, "width": rb.width, "n_entities": rb.n_entities,
            "n_filled": rb.n_cells_filled, "density": round(rb.density, 4),
            "n_collisions": rb.n_collisions, "n_unk": rb.n_unk,
        })

    n = len(grids)
    if n_small_filtered:
        print(f"  [note] dropped {n_small_filtered} blueprint(s) with min(H,W) < {min_dim}")
    if n_unk_filtered:
        print(f"  [note] dropped {n_unk_filtered} blueprint(s) below min_invocab_frac={min_invocab_frac}")
    if n == 0:
        raise RuntimeError("no usable blueprints after rasterization/filtering")

    # Anti-leakage split: assign WHOLE groups (connected components of books that share
    # any identical blueprint; falls back to source_hash, then id) to a single split, so
    # near-duplicate blueprints never straddle train/val/test (the old per-blueprint
    # permutation leaked: one book's siblings landed in both train and test). Targets are
    # counted in BLUEPRINTS (not groups) so the val/test fractions stay honest.
    from collections import defaultdict as _dd
    groups = [m.get("group_id") or m.get("source_hash") or m["id"] for m in metas]
    g2idx = _dd(list)
    for i, g in enumerate(groups):
        g2idx[g].append(i)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    if n >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
    # Size-aware bin-packing: assign LARGEST groups first, each to whichever split is most
    # under-filled relative to its target (ties -> the split with the bigger target). This
    # deterministically routes a giant component into train regardless of seed, instead of
    # relying on it luckily landing there, so val/test stay near their target fractions.
    targets = {"train": n - n_val - n_test, "val": n_val, "test": n_test}
    bins = {"train": [], "val": [], "test": []}
    sizes = {"train": 0, "val": 0, "test": 0}
    rng = np.random.default_rng(seed)
    gkeys = list(g2idx.keys())
    rng.shuffle(gkeys)                                   # seed-varied tie-break
    gkeys.sort(key=lambda g: len(g2idx[g]), reverse=True)  # largest first (stable)
    for g in gkeys:
        s = min(("train", "val", "test"),
                key=lambda k: (sizes[k] / targets[k] if targets[k] > 0 else 9e99, -targets[k]))
        bins[s] += g2idx[g]
        sizes[s] += len(g2idx[g])
    train_idx, val_idx, test_idx = bins["train"], bins["val"], bins["test"]
    print(f"  grouped split over {len(gkeys)} groups "
          f"(largest={max(len(v) for v in g2idx.values())} bp); "
          f"sizes train={sizes['train']} val={sizes['val']} test={sizes['test']}")

    def take(idxs):
        return [grids[i] for i in idxs], [metas[i] for i in idxs]

    splits, split_meta = {}, {}
    for name, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        g, m = take(idxs)
        splits[name] = g
        split_meta[name] = m

    payload = {
        "config": {
            "crop_size": crop_size, "mask_size": mask_size, "max_dim": max_dim,
            "min_dim": min_dim, "min_entities": min_entities, "min_filled": min_filled,
            "min_invocab_frac": min_invocab_frac,
            "val_frac": val_frac, "test_frac": test_frac, "seed": seed,
        },
        "vocab_tokens": vocab.itos,
        "splits": splits,
        "split_meta": split_meta,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Built dataset -> {out_path}")
    print(f"  blueprints (usable): {n}")
    print(f"  split: train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print(f"  crop={crop_size} mask={mask_size} vocab={len(vocab)}")
    return payload


def load_dataset(path: Path) -> dict:
    return torch.load(path, weights_only=False)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _extract_window(grid: np.ndarray, gt: int, gl: int, C: int) -> np.ndarray:
    """Extract a C x C window with top-left at grid coords (gt, gl); pad with EMPTY."""
    H, W = grid.shape
    canvas = np.full((C, C), EMPTY_ID, dtype=np.int16)
    sr0, sr1 = max(0, gt), min(H, gt + C)
    sc0, sc1 = max(0, gl), min(W, gl + C)
    if sr0 < sr1 and sc0 < sc1:
        dr0, dc0 = sr0 - gt, sc0 - gl
        canvas[dr0:dr0 + (sr1 - sr0), dc0:dc0 + (sc1 - sc0)] = grid[sr0:sr1, sc0:sc1]
    return canvas


class FactorioPatchDataset(Dataset):
    """Samples (x, y, mask) patches from a list of blueprint grids.

    train=True  -> each __getitem__ samples a fresh random patch.
    train=False -> patches are deterministic per index (reproducible val/test).
    """

    def __init__(
        self,
        grids: list[np.ndarray],
        crop_size: int = 64,
        mask_size: int = 16,
        train: bool = True,
        length: int | None = None,
        samples_per_bp: int = 16,
        min_nonempty_frac: float = 0.05,
        max_attempts: int = 12,
        seed: int = 0,
    ):
        if mask_size > crop_size:
            raise ValueError("mask_size must be <= crop_size")
        self.grids = grids
        self.C = crop_size
        self.M = mask_size
        self.train = train
        self.min_nonempty_frac = min_nonempty_frac
        self.max_attempts = max_attempts
        self.seed = seed
        # Precompute occupied (r, c) coords per grid for content-biased cropping.
        self.occupied = [np.argwhere(g != EMPTY_ID) for g in grids]
        self.nonempty_grids = [i for i, occ in enumerate(self.occupied) if len(occ) > 0]
        if not self.nonempty_grids:
            raise RuntimeError("all grids are empty")
        if length is not None:
            self.length = length
        else:
            self.length = max(len(grids) * samples_per_bp, 256)
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.length

    def _pick_grid(self, rng) -> int:
        return int(self.nonempty_grids[rng.integers(len(self.nonempty_grids))])

    def _make_sample(self, gi: int, rng):
        grid = self.grids[gi]
        occ = self.occupied[gi]
        C, M = self.C, self.M
        best = None      # highest mask non-empty frac seen (fallback)
        best_ctx = None  # highest frac among attempts with adequate context
        for _ in range(self.max_attempts):
            # Content-biased crop: anchor on a random occupied cell, jitter its
            # position within the window.
            ar, ac = occ[rng.integers(len(occ))]
            gt = int(ar) - int(rng.integers(0, C))
            gl = int(ac) - int(rng.integers(0, C))
            canvas = _extract_window(grid, gt, gl, C)
            cocc = np.argwhere(canvas != EMPTY_ID)
            if len(cocc) == 0:
                continue
            # Mask anchored to cover an occupied cell.
            cr, cc = cocc[rng.integers(len(cocc))]
            lo_t, hi_t = max(0, int(cr) - M + 1), min(C - M, int(cr))
            lo_l, hi_l = max(0, int(cc) - M + 1), min(C - M, int(cc))
            mt = int(rng.integers(lo_t, hi_t + 1)) if hi_t >= lo_t else 0
            ml = int(rng.integers(lo_l, hi_l + 1)) if hi_l >= lo_l else 0
            region = canvas[mt:mt + M, ml:ml + M]
            n_mask_nonempty = int((region != EMPTY_ID).sum())
            n_total_nonempty = len(cocc)
            n_context = n_total_nonempty - n_mask_nonempty
            # Require meaningful surrounding context, not just one stray cell.
            min_context = max(1, int(0.15 * n_total_nonempty))
            frac = n_mask_nonempty / (M * M)
            if best is None or frac > best[0]:
                best = (frac, canvas, mt, ml)
            if n_context >= min_context and (best_ctx is None or frac > best_ctx[0]):
                best_ctx = (frac, canvas, mt, ml)
            if frac >= self.min_nonempty_frac and n_context >= min_context:
                best_ctx = (frac, canvas, mt, ml)
                break

        _, canvas, mt, ml = best_ctx if best_ctx is not None else best
        y = canvas.astype(np.int64)
        x = y.copy()
        mask = np.zeros((C, C), dtype=bool)
        mask[mt:mt + M, ml:ml + M] = True
        x[mask] = MASK_ID
        return {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(y),
            "mask": torch.from_numpy(mask),
        }

    def __getitem__(self, idx: int):
        if self.train:
            rng = self._rng
            gi = self._pick_grid(rng)
        else:
            rng = np.random.default_rng(self.seed + idx + 1)
            gi = self.nonempty_grids[idx % len(self.nonempty_grids)]
        return self._make_sample(gi, rng)


def make_datasets(payload: dict, train_length: int | None = None,
                  val_length: int | None = None, seed: int = 0):
    """Build train/val/test FactorioPatchDataset objects + Vocab from a payload."""
    cfg = payload["config"]
    C, M = cfg["crop_size"], cfg["mask_size"]
    vocab = Vocab(payload["vocab_tokens"])
    sp = payload["splits"]

    def mk(name, train, length):
        grids = sp[name]
        if not grids:
            return None
        return FactorioPatchDataset(grids, crop_size=C, mask_size=M, train=train,
                                    length=length, seed=seed)

    n_val_default = max(len(sp.get("val") or []) * 8, 128)
    n_test_default = max(len(sp.get("test") or []) * 8, 128)
    return {
        "train": mk("train", True, train_length),
        "val": mk("val", False, val_length or n_val_default),
        "test": mk("test", False, n_test_default),
        "vocab": vocab,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the masked-patch dataset (.pt) from blueprints.jsonl + vocab.")
    ap.add_argument("--blueprints", type=Path, required=True)
    ap.add_argument("--vocab", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed/dataset.pt"))
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--mask-size", type=int, default=16)
    ap.add_argument("--max-dim", type=int, default=512)
    ap.add_argument("--min-dim", type=int, default=0,
                    help="drop blueprints smaller than this in either dimension (keeps context around the mask)")
    ap.add_argument("--min-entities", type=int, default=4)
    ap.add_argument("--min-invocab-frac", type=float, default=0.0,
                    help="drop blueprints whose occupied cells are mostly UNK")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    vocab = Vocab.load(args.vocab)
    build_dataset(
        args.blueprints, vocab, args.out,
        crop_size=args.crop_size, mask_size=args.mask_size, max_dim=args.max_dim,
        min_dim=args.min_dim, min_entities=args.min_entities, min_invocab_frac=args.min_invocab_frac,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
