"""Dihedral (D4) symmetry for Factorio token grids.

Factorio blueprints are heavily rotation/reflection-structured, so a model that sees them
from all 8 D4 views (4 rotations × 2 reflections) gets ~8× effective data (train-time
augmentation) and a free ensemble at inference (test-time augmentation). A transform both
moves the grid spatially AND remaps every entity's direction (2.0 is 16-way: N=0, E=4,
S=8, W=12, + diagonals).
"""

from __future__ import annotations

import numpy as np
import torch

from .vocab import split_token

NDIR = 16
# the 8 D4 elements as (rot_k quarter-turns CCW, horizontal-flip); spatial = fliplr?(rot90(g,k))
D4 = [(k, f) for f in (False, True) for k in range(4)]


def _dir_after(d: int, rot_k: int, flip: bool) -> int:
    d = (d - 4 * rot_k) % NDIR          # np.rot90 CCW maps 'up'->'left' => d-4 per turn
    return (-d) % NDIR if flip else d   # fliplr mirrors E<->W, keeps N/S


def build_perms(vocab):
    """For each D4 element, a token-id permutation: id -> id of the SAME entity (name+io)
    with its direction transformed. Tokens whose transformed direction isn't in the vocab
    keep their id (safe fallback so TTA never invents tokens)."""
    perms = []
    for rot_k, flip in D4:
        perm = np.arange(len(vocab), dtype=np.int64)
        for i, tok in enumerate(vocab.itos):
            name, d = split_token(tok)      # name may carry :io; d is None for specials
            if d is None:
                continue
            perm[i] = vocab.stoi.get(f"{name}|{_dir_after(int(d), rot_k, flip)}", i)
        perms.append(perm)
    return perms


def transform_grid(grid: np.ndarray, rot_k: int, flip: bool, perm: np.ndarray) -> np.ndarray:
    """Apply a D4 element to a token-id grid: spatial transform + direction remap."""
    g = np.rot90(grid, rot_k)
    if flip:
        g = np.fliplr(g)
    return perm[g]


@torch.no_grad()
def tta_logits(model, x, vocab, device, perms_t=None):
    """Test-time augmentation: predict on all 8 D4 views, un-transform each back to the
    original frame (spatial inverse + vocab permute), average the probabilities. Returns
    [B, V, H, W] averaged probs (argmax = the TTA prediction)."""
    if perms_t is None:
        perms_t = [torch.as_tensor(p, device=device) for p in build_perms(vocab)]
    acc = None
    for (rot_k, flip), perm in zip(D4, perms_t):
        xg = torch.rot90(x, rot_k, dims=(-2, -1))
        if flip:
            xg = torch.flip(xg, dims=(-1,))
        xg = perm[xg]                                   # remap directions
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            p = model(xg).float().softmax(dim=1)        # [B,V,H,W] probs
        p = p[:, perm, :, :]                            # vocab back to original-token order
        if flip:                                        # spatial inverse: undo flip, then rot
            p = torch.flip(p, dims=(-1,))
        p = torch.rot90(p, -rot_k, dims=(-2, -1))
        acc = p if acc is None else acc + p
    return acc / len(D4)
