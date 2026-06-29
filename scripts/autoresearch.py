#!/usr/bin/env python
"""AutoResearch harness for Factorio patch-inpainting.

Evaluate a config (one or an ensemble of checkpoints, optional D4 test-time augmentation,
optional MaskGIT iterative decode) on val + test, compute entity-token accuracy, and append
to a running ledger (with a Modal-cost estimate). The search metric is VAL entity-acc
(mirroring the brain2qwerty AutoResearch protocol); TEST is reported for confirmation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from factorio_patches.dataset import FactorioPatchDataset, load_dataset
from factorio_patches.eval import load_checkpoint
from factorio_patches.metrics import maskgit_decode
from factorio_patches.symmetry import build_perms, tta_logits
from factorio_patches.vocab import EMPTY_ID, Vocab

LEDGER = Path("outputs/autoresearch/experiments.jsonl")
A10G_RATE = 1.10  # $/hr (Modal A10G); A100-40GB ~2.50


def load_models(paths, device):
    models, vocab = [], None
    for p in paths:
        m, vocab, _ = load_checkpoint(Path(p), device)
        m.eval()
        models.append(m)
    return models, vocab


@torch.no_grad()
def predict_probs(models, x, mask, vocab, device, tta=False, maskgit=0, perms_t=None):
    """Averaged probabilities [B,V,H,W] over the ensemble (and D4 views if tta)."""
    accs = []
    for m in models:
        if maskgit:
            pred = maskgit_decode(m, x, mask, steps=maskgit)
            accs.append(torch.nn.functional.one_hot(pred, len(vocab)).permute(0, 3, 1, 2).float())
        elif tta:
            accs.append(tta_logits(m, x, vocab, device, perms_t))
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                accs.append(m(x).float().softmax(dim=1))
    return torch.stack(accs).mean(0)


@torch.no_grad()
def entity_acc(models, grids, vocab, device, tta=False, maskgit=0, n=1500, bs=128, seed=0):
    """Exact entity-token accuracy over masked, non-empty-target cells (matches metrics.py)."""
    ds = FactorioPatchDataset(grids, train=False, length=n, seed=seed)
    perms_t = [torch.as_tensor(p, device=device) for p in build_perms(vocab)] if tta else None
    tp = tot = 0
    for b in DataLoader(ds, batch_size=bs):
        x, y, mask = b["x"].to(device), b["y"].to(device), b["mask"].to(device)
        pred = predict_probs(models, x, mask, vocab, device, tta, maskgit, perms_t).argmax(1)
        m = mask & (y != EMPTY_ID)
        tp += int((pred[m] == y[m]).sum()); tot += int(m.sum())
    return tp / max(tot, 1)


def log_experiment(name, val_acc, test_acc=None, est_cost=0.0, kept=None, note="", group="A"):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    exp = len(rows)
    best = max([r["val_acc"] for r in rows], default=-1)
    if kept is None:
        kept = val_acc > best
    rec = {"exp": exp, "name": name, "val_acc": round(val_acc, 4),
           "test_acc": round(test_acc, 4) if test_acc is not None else None,
           "running_best_val": round(max(val_acc, best), 4), "kept": bool(kept),
           "est_cost": round(est_cost, 3), "group": group, "note": note, "ts": int(time.time())}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    total_cost = sum(r.get("est_cost", 0) for r in rows) + est_cost
    star = " *BEST*" if kept else ""
    print(f"[exp {exp:>2}] {name:<34} val={val_acc:.4f} "
          f"test={test_acc if test_acc is None else round(test_acc,4)} "
          f"best={max(val_acc,best):.4f}{star}  spend=${total_cost:.2f}")
    return rec
