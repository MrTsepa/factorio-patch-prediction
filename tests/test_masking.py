import numpy as np
import torch

from factorio_patches.dataset import FactorioPatchDataset, _extract_window
from factorio_patches.vocab import EMPTY_ID, MASK_ID


def make_grid(H=40, W=40, fill_token=5, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    g = np.full((H, W), EMPTY_ID, dtype=np.int16)
    occ = rng.random((H, W)) < density
    g[occ] = fill_token
    return g


def test_window_padding():
    g = np.arange(9, dtype=np.int16).reshape(3, 3) + 3  # values 3..11, none EMPTY
    # window beyond bounds pads with EMPTY
    w = _extract_window(g, -1, -1, 5)
    assert w.shape == (5, 5)
    assert w[0, 0] == EMPTY_ID  # padded
    assert w[1, 1] == g[0, 0]   # aligned


def test_item_shapes_and_mask():
    grids = [make_grid()]
    ds = FactorioPatchDataset(grids, crop_size=64, mask_size=16, train=True, length=10, seed=1)
    s = ds[0]
    assert s["x"].shape == (64, 64)
    assert s["y"].shape == (64, 64)
    assert s["mask"].shape == (64, 64)
    assert s["mask"].dtype == torch.bool
    # exactly one 16x16 block masked
    assert int(s["mask"].sum()) == 16 * 16


def test_input_is_masked_target_is_not():
    grids = [make_grid()]
    ds = FactorioPatchDataset(grids, crop_size=64, mask_size=16, train=True, length=10, seed=2)
    s = ds[0]
    x, y, m = s["x"], s["y"], s["mask"]
    # inside mask, x == MASK; outside, x == y
    assert torch.all(x[m] == MASK_ID)
    assert torch.all(x[~m] == y[~m])
    # target never contains the MASK token
    assert int((y == MASK_ID).sum()) == 0


def test_mask_covers_nonempty_when_possible():
    grids = [make_grid(density=0.4, seed=3)]
    ds = FactorioPatchDataset(grids, crop_size=64, mask_size=16, train=True,
                              length=20, min_nonempty_frac=0.05, seed=3)
    # With a dense grid, every sampled mask should contain some non-empty target.
    ok = 0
    for i in range(20):
        s = ds[i]
        region = s["y"][s["mask"]]
        if int((region != EMPTY_ID).sum()) > 0:
            ok += 1
    assert ok == 20


def test_eval_mode_is_deterministic():
    grids = [make_grid(seed=4), make_grid(seed=5)]
    ds = FactorioPatchDataset(grids, crop_size=64, mask_size=16, train=False, length=8, seed=7)
    a = ds[3]
    b = ds[3]
    assert torch.equal(a["x"], b["x"])
    assert torch.equal(a["y"], b["y"])
    assert torch.equal(a["mask"], b["mask"])
