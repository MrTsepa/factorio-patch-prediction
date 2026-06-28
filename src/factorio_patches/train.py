"""Train the U-Net patch-inpainting model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # noqa: F401  (so torch._dynamo.config is reachable for --compile)
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import build_dataset, load_dataset, make_datasets
from .metrics import evaluate, format_metrics, masked_ce_loss, most_common_entity_id
from .model import build_model
from .render import render_prediction_set
from .vocab import EMPTY_ID, Vocab


def pick_device(arg: str) -> torch.device:
    if arg and arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dataset(data: Path, crop: int, mask: int, max_dim: int, seed: int) -> Path:
    """Return a dataset.pt path, building it from blueprints.jsonl + vocab.json if needed."""
    data = Path(data)
    if data.is_file():
        return data
    ds_path = data / "dataset.pt"
    if ds_path.exists():
        return ds_path
    bp = data / "blueprints.jsonl"
    vp = data / "vocab.json"
    if not (bp.exists() and vp.exists()):
        raise FileNotFoundError(
            f"no dataset.pt and missing {bp} or {vp}; run extract + vocab first")
    print(f"dataset.pt not found; building from {bp} + {vp} ...")
    build_dataset(bp, Vocab.load(vp), ds_path, crop_size=crop, mask_size=mask,
                  max_dim=max_dim, seed=seed)
    return ds_path


def render_epoch_samples(model, val_ds, vocab, device, out_dir: Path, n: int = 6):
    if val_ds is None:
        return
    model.eval()
    n = min(n, len(val_ds))
    with torch.no_grad():
        for i in range(n):
            s = val_ds[i]
            x = s["x"].unsqueeze(0).to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)[0].cpu().numpy()
            y = s["y"].numpy()
            mask = s["mask"].numpy()
            xin = s["x"].numpy()
            # show ground truth tokens outside the mask in the prediction panel
            pred_full = y.copy()
            pred_full[mask] = pred[mask]
            render_prediction_set(xin, y, pred_full, mask, vocab, out_dir, f"val_{i:02d}")


def _pred_panel(model, val_ds, vocab, device, n=4):
    """A stacked input|target|prediction|diff panel of n val examples, for wandb."""
    from PIL import Image

    from .render import hconcat, render_diff, render_grid
    rows = []
    model.eval()
    with torch.no_grad():
        for i in range(min(n, len(val_ds))):
            s = val_ds[i]
            pred = model(s["x"].unsqueeze(0).to(device)).argmax(dim=1)[0].cpu().numpy()
            y, m, xin = s["y"].numpy(), s["mask"].numpy(), s["x"].numpy()
            pf = y.copy(); pf[m] = pred[m]
            rows.append(hconcat([render_grid(xin, vocab, cell=8, mask=m),
                                 render_grid(y, vocab, cell=8, mask=m),
                                 render_grid(pf, vocab, cell=8, mask=m),
                                 render_diff(y, pf, m, cell=8)]))
    w = max(r.width for r in rows)
    h = sum(r.height for r in rows) + 6 * (len(rows) - 1)
    out = Image.new("RGB", (w, h), (20, 20, 22))
    yo = 0
    for r in rows:
        out.paste(r, (0, yo)); yo += r.height + 6
    return out


def _seed_worker(_worker_id):
    """Give each DataLoader worker an independent, per-epoch RNG stream.

    Without this, num_workers>0 forks copies of the dataset's single numpy Generator, so
    every worker emits IDENTICAL patches and re-forks the SAME set each epoch (PyTorch
    reseeds python/torch + numpy's *global* RNG per worker, but NOT a Generator instance).
    ``info.seed`` = base_seed + worker_id, and PyTorch draws a fresh base_seed each epoch
    for non-persistent workers, so this varies per worker AND per epoch.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(info.seed % (2 ** 32))


def train(args, on_epoch_end=None) -> dict:
    """Train; ``on_epoch_end()`` (if given) is called after each epoch's checkpoint save —
    used on Modal to vol.commit() so a mid-run crash doesn't lose all checkpoints."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ds_path = resolve_dataset(Path(args.data), args.crop_size, args.mask_size, args.max_dim, args.seed)
    payload = load_dataset(ds_path)
    cfg = payload["config"]
    # Honor the dataset's crop/mask (the model is conv so it adapts, but keep them aligned).
    mg = getattr(args, "maskgit", False)
    ds = make_datasets(payload, train_length=args.train_samples, val_length=args.val_samples,
                       seed=args.seed, size_power=getattr(args, "size_power", 0.0), maskgit=mg)
    eval_steps = args.maskgit_steps if mg else 0
    if mg:
        print(f"MaskGIT: variable-ratio masked training + {eval_steps}-step iterative decode eval")
    vocab = ds["vocab"]
    prior_id = most_common_entity_id(payload["splits"]["train"])

    print(f"device={device} vocab={len(vocab)} crop={cfg['crop_size']} mask={cfg['mask_size']}")
    print(f"split sizes: train={len(payload['splits']['train'])} "
          f"val={len(payload['splits']['val'])} test={len(payload['splits']['test'])} blueprints")

    train_loader = DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True,
                              worker_init_fn=_seed_worker)
    val_loader = DataLoader(ds["val"], batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers) if ds["val"] else None
    test_loader = DataLoader(ds["test"], batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if ds["test"] else None

    model = build_model(args.arch, len(vocab), d_model=args.d_model,
                        depth=args.depth, heads=args.heads, patch=args.patch).to(device)
    print(f"arch={args.arch} model params: {model.num_params():,}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    # bf16 autocast (Tensor Cores) + torch.compile speed up the GPU forward a lot,
    # especially the transformer's attention; no-ops on CPU/MPS. Keep the original
    # `model` for checkpoints/eval; compile only the training-forward `fwd` (shares
    # params, but avoids the compiled state_dict's "_orig_mod." key prefix).
    use_amp = args.amp == "on" or (args.amp == "auto" and device.type == "cuda")
    fwd = model
    if args.compile and device.type == "cuda":
        try:
            torch._dynamo.config.suppress_errors = True  # fall back to eager, never crash
            fwd = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as e:  # pragma: no cover - environment dependent
            print(f"[warn] torch.compile failed ({e}); continuing uncompiled")
    if use_amp:
        print(f"AMP: bfloat16 autocast on {device.type}")

    # Down-weight EMPTY in the loss to fight the heavy class imbalance.
    class_weight = torch.ones(len(vocab), device=device)
    class_weight[EMPTY_ID] = args.empty_weight

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb, name=args.run_name,
                         group=getattr(args, "wandb_group", None),
                         tags=getattr(args, "wandb_tags", None), config={
            "arch": args.arch, "d_model": args.d_model, "depth": args.depth,
            "heads": args.heads, "patch": args.patch, "params": model.num_params(),
            "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
            "weight_decay": args.weight_decay, "empty_weight": args.empty_weight,
            "crop": cfg["crop_size"], "mask": cfg["mask_size"], "vocab": len(vocab),
            "train_blueprints": len(payload["splits"]["train"]), "dataset": str(ds_path)})

    history = []
    best_metric = -1.0
    best_epoch = 0
    no_improve = 0
    epoch_times: list[float] = []
    stop_reason = None
    metrics_path = out / "metrics.jsonl"
    metrics_path.write_text("")
    t_start = time.time()  # wall-clock budget starts after data/model/compile setup

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        running_gn = 0.0
        nb = 0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = fwd(x)
                loss = masked_ce_loss(logits, y, mask, weight=class_weight)
            opt.zero_grad()
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
            running_gn += float(gn)
            nb += 1
            pbar.set_postfix(loss=f"{running / nb:.3f}")
        train_loss = running / max(nb, 1)
        grad_norm = running_gn / max(nb, 1)
        train_dt = time.time() - t0
        samples_per_sec = (nb * args.batch_size) / max(train_dt, 1e-6)
        sched.step()

        val_metrics = evaluate(model, val_loader, device, prior_id, vocab=vocab,
                               maskgit_steps=eval_steps) if val_loader else None
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": train_loss, "seconds": round(dt, 1),
               "val": val_metrics}
        history.append(rec)
        with metrics_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"\nepoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  ({dt:.1f}s)")
        if val_metrics:
            print(format_metrics("val", val_metrics))
        if val_metrics and "model_singleshot" in val_metrics:
            ss = val_metrics["model_singleshot"]; it = val_metrics["model"]
            print(f"[val] single-shot : entity_acc={ss['entity_token_acc']:.3f} "
                  f"nonempty_f1={ss['non_empty_f1']:.3f}  "
                  f"(iterative-decode gain {it['entity_token_acc'] - ss['entity_token_acc']:+.3f})")

        # checkpoints (last.pt every epoch; best.pt on val improvement below)
        ckpt = {"model_state": model.state_dict(), "vocab_tokens": vocab.itos,
                "config": {"crop_size": cfg["crop_size"], "mask_size": cfg["mask_size"],
                           "arch": args.arch, "d_model": args.d_model,
                           "depth": args.depth, "heads": args.heads, "patch": args.patch},
                "epoch": epoch,
                "val_metrics": val_metrics, "prior_id": prior_id, "dataset": str(ds_path)}
        torch.save(ckpt, out / "last.pt")

        # --- early stopping: monitor val entity_token_acc with a min-delta ---
        crit = val_metrics["model"]["entity_token_acc"] if val_metrics else -1.0
        if crit > best_metric + args.min_delta:
            best_metric, best_epoch, no_improve = crit, epoch, 0
            torch.save(ckpt, out / "best.pt")
            print(f"  -> new best (val entity_token_acc={crit:.3f}) saved to best.pt")
        else:
            no_improve += 1
            print(f"  -> no improvement ({no_improve}/{args.patience}); "
                  f"best={best_metric:.3f} @ epoch {best_epoch}")

        # --- stop conditions (evaluated AFTER best.pt is saved so it is always current) ---
        epoch_times.append(dt)
        elapsed = time.time() - t_start
        if args.patience > 0 and no_improve >= args.patience:
            stop_reason = (f"no val improvement in {no_improve} epochs "
                           f"(best={best_metric:.3f} @ epoch {best_epoch})")
        elif args.max_seconds > 0 and elapsed + 1.15 * max(epoch_times) >= args.max_seconds:
            stop_reason = (f"wall-clock cap: {elapsed:.0f}s elapsed; next epoch "
                           f"(~{max(epoch_times):.0f}s) would exceed {args.max_seconds:.0f}s budget")

        if run and val_metrics:
            import wandb
            log = {"epoch": epoch, "train/loss": train_loss, "train/lr": sched.get_last_lr()[0],
                   "train/grad_norm": grad_norm, "train/epoch_time_s": train_dt,
                   "train/samples_per_sec": samples_per_sec,
                   "early_stop/no_improve": no_improve,
                   "early_stop/best_val_entity_acc": best_metric,
                   "time/elapsed_s": round(elapsed, 1)}
            # val/* includes io_acc (evaluate() is called with vocab) -> logs val/io_acc
            log.update({f"val/{k}": v for k, v in val_metrics["model"].items()
                        if isinstance(v, (int, float))})
            if "model_singleshot" in val_metrics:   # 2x2: isolate decoding vs training-scheme
                log.update({f"val_ss/{k}": v for k, v in val_metrics["model_singleshot"].items()
                            if isinstance(v, (int, float))})
            md_, bm_ = val_metrics["model"], val_metrics["baseline_majority_entity"]
            log.update({f"val_baseline/{k}": bm_[k] for k in ("entity_token_acc", "non_empty_f1")})
            # rising "lift over majority baseline" is more informative than the flat line
            log["val/entity_acc_lift"] = md_["entity_token_acc"] - bm_["entity_token_acc"]
            log["val/f1_lift"] = md_["non_empty_f1"] - bm_["non_empty_f1"]
            if epoch % max(1, args.epochs // 5) == 0 or stop_reason or epoch == args.epochs:
                log["val/predictions"] = wandb.Image(_pred_panel(model, ds["val"], vocab, device))
            run.log(log)

        render_epoch_samples(model, ds["val"], vocab, device, out / "preds" / f"epoch_{epoch:02d}", n=6)

        if on_epoch_end is not None:
            on_epoch_end()      # e.g. vol.commit() on Modal -> checkpoints survive a crash

        if stop_reason:
            print(f"\n[early-stop] stopping after epoch {epoch}: {stop_reason}")
            break

    # --- restore best weights (early stopping) before the final test eval ---
    best_path = out / "best.pt"
    if args.restore_best and best_path.exists():
        best = torch.load(best_path, weights_only=False)
        model.load_state_dict(best["model_state"])
        print(f"\nrestored best weights from epoch {best['epoch']} "
              f"(val entity_token_acc={best_metric:.3f})")
    elif not best_path.exists():
        print("\n[warn] no best.pt (no validation split?); keeping last weights for test")

    # Final test evaluation with the (restored) best checkpoint.
    summary = {"best_val_entity_token_acc": best_metric, "best_epoch": best_epoch,
               "epochs_run": len(epoch_times), "epochs_cap": args.epochs,
               "stopped_early": stop_reason is not None, "stop_reason": stop_reason,
               "wall_clock_s": round(time.time() - t_start, 1),
               "device": str(device), "model_params": model.num_params(),
               "empty_weight": args.empty_weight, "lr": args.lr, "d_model": args.d_model,
               "crop_size": cfg["crop_size"], "mask_size": cfg["mask_size"]}
    if test_loader:
        test_metrics = evaluate(model, test_loader, device, prior_id, vocab=vocab,
                                maskgit_steps=eval_steps)
        print("\n=== TEST (best checkpoint) ===")
        print(format_metrics("test", test_metrics))
        summary["test"] = test_metrics
        if run:
            run.log({f"test/{k}": v for k, v in test_metrics["model"].items()
                     if isinstance(v, (int, float))})
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if run:
        run.summary.update({"best_val_entity_token_acc": best_metric})
        run.finish()
    print(f"\nSaved checkpoints + metrics to {out}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the patch-inpainting U-Net.")
    ap.add_argument("--data", type=Path, default=Path("data/processed"),
                    help="dir with blueprints.jsonl+vocab.json (auto-builds dataset.pt) or a dataset.pt file")
    ap.add_argument("--out", type=Path, default=Path("runs/poc_001"))
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--mask-size", type=int, default=16)
    ap.add_argument("--max-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--wandb", default=None, help="wandb project to log to (disabled if unset)")
    ap.add_argument("--run-name", default=None, help="wandb run name (make it unique + descriptive)")
    ap.add_argument("--wandb-group", default=None, help="wandb group (e.g. one launch id)")
    ap.add_argument("--wandb-tags", nargs="*", default=None, help="wandb tags for filtering")
    ap.add_argument("--amp", choices=["auto", "on", "off"], default="auto",
                    help="bfloat16 autocast (auto = on for CUDA)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA)")
    ap.add_argument("--empty-weight", type=float, default=0.2,
                    help="loss weight for the EMPTY class (down-weight to fight imbalance)")
    ap.add_argument("--arch", choices=["unet", "transformer"], default="unet")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--depth", type=int, default=6, help="transformer encoder layers")
    ap.add_argument("--heads", type=int, default=6, help="transformer attention heads")
    ap.add_argument("--patch", type=int, default=2, help="transformer patchify stride (power of 2)")
    ap.add_argument("--size-power", type=float, default=0.0,
                    help="mild size-weighting of TRAIN blueprint sampling (0=uniform, 0.5=sqrt)")
    ap.add_argument("--maskgit", action="store_true",
                    help="MaskGIT: variable-ratio masked training + iterative-decode eval")
    ap.add_argument("--maskgit-steps", type=int, default=8, help="iterative decode steps")
    ap.add_argument("--train-samples", type=int, default=4000, help="patches sampled per epoch")
    ap.add_argument("--val-samples", type=int, default=1000)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    # --- early stopping + wall-clock budget ---
    ap.add_argument("--patience", type=int, default=8,
                    help="early-stop after this many epochs with no val improvement (<=0 disables)")
    ap.add_argument("--min-delta", type=float, default=1e-3,
                    help="min val entity_token_acc gain that counts as an improvement")
    ap.add_argument("--max-seconds", type=float, default=0.0,
                    help="hard wall-clock cap (s) for the training loop; stops BEFORE an epoch "
                         "that would exceed it, so the run never overruns (0 = no cap)")
    ap.add_argument("--restore-best", dest="restore_best", action="store_true", default=True,
                    help="reload best.pt weights at the end of training (default: on)")
    ap.add_argument("--no-restore-best", dest="restore_best", action="store_false")
    args = ap.parse_args(argv)
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
