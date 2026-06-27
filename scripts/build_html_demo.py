#!/usr/bin/env python
"""Build a self-contained HTML demo of a trained model's inpainting results.

For each held-out test crop it shows the abstract input/target/prediction/diff
panel AND a game-accurate FBSR render of the target vs. the model's prediction.
Images are base64-embedded so the page is a single portable file.

  uv run python scripts/build_html_demo.py --checkpoint runs/poc_002/best.pt \
      --out outputs/demo_2.0 --num 6
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from factorio_patches.blueprint_decode import encode_blueprint_string
from factorio_patches.dataset import load_dataset, make_datasets
from factorio_patches.eval import VERSION_2_0, grid_to_blueprint, load_checkpoint
from factorio_patches.metrics import evaluate, most_common_entity_id
from factorio_patches.render import hconcat, render_diff, render_grid
from factorio_patches.vocab import EMPTY_ID


def b64(img, max_w=None, fmt="PNG"):
    if max_w and img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    if fmt == "JPEG" and img.mode == "RGBA":     # flatten transparency onto the game bg
        bg = Image.new("RGB", img.size, (27, 27, 29))
        bg.paste(img, mask=img.split()[3])
        img = bg
    buf = io.BytesIO()
    img.save(buf, fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def fbsr_render(jobs, fbsr_run):
    if not jobs:
        return
    cmds = "".join(f"bot-render -f={t} -o={p} -full\n" for t, p in jobs) + "exit\n"
    try:
        subprocess.run([fbsr_run], input=cmds, text=True, capture_output=True, timeout=900)
    except Exception as e:
        print(f"[warn] FBSR render failed ({e}); HTML will show abstract panels only")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/demo_2.0"))
    ap.add_argument("--num", type=int, default=6)
    ap.add_argument("--split", default="test")
    ap.add_argument("--fbsr-run", default=str(Path(__file__).resolve().parent / "fbsr.sh"))
    ap.add_argument("--version", type=int, default=VERSION_2_0)
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)
    cfg = ckpt["config"]
    payload = load_dataset(args.data if args.data else Path(ckpt["dataset"]))
    ds = make_datasets(payload)
    split = ds[args.split]
    prior = ckpt.get("prior_id") or most_common_entity_id(payload["splits"]["train"])
    metrics = evaluate(model, DataLoader(split, batch_size=64), device, prior)

    # pick content-rich, well-reconstructed examples
    cand = []
    with torch.no_grad():
        for i in range(len(split)):
            s = split[i]
            y, m = s["y"].numpy(), s["mask"].numpy()
            nz = int((y[m] != EMPTY_ID).sum())
            if nz < 25:
                continue
            pred = model(s["x"].unsqueeze(0)).argmax(1)[0].numpy()
            cand.append((float((pred[m] == y[m]).mean()), nz, i))
    cand.sort(reverse=True)
    picks = [i for _, _, i in cand[:args.num]] or list(range(min(args.num, len(split))))

    tmp = Path("/tmp/html_demo"); tmp.mkdir(parents=True, exist_ok=True)
    jobs, examples = [], []
    with torch.no_grad():
        for rank, i in enumerate(picks):
            s = split[i]; mask = s["mask"].numpy(); y = s["y"].numpy(); xin = s["x"].numpy()
            pred = model(s["x"].unsqueeze(0)).argmax(1)[0].numpy()
            pf = y.copy(); pf[mask] = pred[mask]
            acc = float((pred[mask] == y[mask]).mean())
            panel = hconcat([render_grid(xin, vocab, cell=9, mask=mask),
                             render_grid(y, vocab, cell=9, mask=mask),
                             render_grid(pf, vocab, cell=9, mask=mask),
                             render_diff(y, pf, mask, cell=9)])
            for tag, g in (("target", y), ("prediction", pf)):
                bp = grid_to_blueprint(g, vocab, label=f"ex{i} {tag}", version=args.version)
                (tmp / f"{rank}_{tag}.txt").write_text(encode_blueprint_string(bp))
                jobs.append((str((tmp / f"{rank}_{tag}.txt").resolve()),
                             str((tmp / f"{rank}_{tag}.png").resolve())))
            examples.append({"i": i, "rank": rank, "acc": acc, "panel": panel,
                             "masked_nonempty": int((y[mask] != EMPTY_ID).sum())})

    print(f"rendering {len(jobs)} blueprints via FBSR ...")
    fbsr_render(jobs, args.fbsr_run)

    args.out.mkdir(parents=True, exist_ok=True)
    html = _html(model, vocab, cfg, metrics, examples, tmp, payload)
    (args.out / "index.html").write_text(html)
    print(f"wrote demo -> {args.out / 'index.html'}  ({len(html)//1024} KB)")
    return 0


def _metric_rows(metrics):
    md, be, bp = metrics["model"], metrics["baseline_empty"], metrics["baseline_majority_entity"]
    rows = [
        ("non-empty F1 (detection)", md["non_empty_f1"], be["non_empty_f1"], bp["non_empty_f1"]),
        ("entity-token accuracy (exact)", md["entity_token_acc"], be["entity_token_acc"], bp["entity_token_acc"]),
        ("top-5 accuracy", md.get("top5_acc", 0), None, None),
        ("masked-cell accuracy", md["masked_acc"], be["masked_acc"], bp["masked_acc"]),
    ]
    out = ""
    for name, m, e, p in rows:
        cells = f"<td class=hl>{m:.3f}</td>"
        cells += f"<td>{e:.3f}</td>" if e is not None else "<td>—</td>"
        cells += f"<td>{p:.3f}</td>" if p is not None else "<td>—</td>"
        out += f"<tr><td>{name}</td>{cells}</tr>"
    return out


def _html(model, vocab, cfg, metrics, examples, tmp, payload):
    n_train = len(payload["splits"]["train"])
    n_val = len(payload["splits"]["val"]); n_test = len(payload["splits"]["test"])
    arch = cfg.get("arch", "unet")
    desc = (f"transformer (ViT) · d_model {cfg.get('d_model')} · depth {cfg.get('depth')} · "
            f"heads {cfg.get('heads')} · patch {cfg.get('patch')}") if arch == "transformer" \
        else f"U-Net · d_model {cfg.get('d_model')}"
    cards = ""
    for ex in examples:
        rank = ex["rank"]
        gt = tmp / f"{rank}_target.png"; pr = tmp / f"{rank}_prediction.png"
        game = ""
        if gt.exists() and pr.exists():
            game = (
                '<div class=game>'
                f'<figure><img src="{b64(Image.open(gt), 760, "JPEG")}">'
                '<figcaption>TARGET — game render (FBSR)</figcaption></figure>'
                f'<figure><img src="{b64(Image.open(pr), 760, "JPEG")}">'
                '<figcaption>PREDICTION — model fills the hole</figcaption></figure>'
                '</div>')
        cards += (
            f'<div class=ex><div class=exhead>Example {ex["i"]} · masked cells with '
            f'entities: {ex["masked_nonempty"]} · in-hole accuracy '
            f'<b>{ex["acc"]:.0%}</b></div>'
            f'<img class=abs src="{b64(ex["panel"], 1100)}">'
            '<div class=lbl>input (16×16 hole) · target · prediction · diff (green=correct)</div>'
            f'{game}</div>')

    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Factorio patch inpainting — native 2.0 ({arch})</title>
<style>
 body{{background:#1b1b1d;color:#e8e8e8;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:32px;}}
 h1{{margin:0 0 4px;font-size:26px}} .sub{{color:#9aa;margin-bottom:20px}}
 .card{{background:#242427;border:1px solid #333;border-radius:10px;padding:16px 20px;margin-bottom:22px}}
 table{{border-collapse:collapse;margin-top:6px}} td,th{{padding:5px 14px;text-align:right;border-bottom:1px solid #333}}
 td:first-child,th:first-child{{text-align:left;color:#bcd}} .hl{{color:#6f6;font-weight:700}}
 .ex{{background:#242427;border:1px solid #333;border-radius:10px;padding:14px;margin-bottom:20px}}
 .exhead{{color:#cde;margin-bottom:8px}} img.abs{{width:100%;max-width:1100px;image-rendering:pixelated;border-radius:6px}}
 .lbl{{color:#889;font-size:12px;margin:4px 0 10px}}
 .game{{display:flex;gap:14px;flex-wrap:wrap}} .game figure{{margin:0;flex:1;min-width:320px}}
 .game img{{width:100%;border-radius:6px}} figcaption{{color:#9aa;font-size:12px;margin-top:4px}}
 b{{color:#fff}} code{{color:#fc9}}
</style></head><body>
<h1>Factorio blueprint patch inpainting — native Factorio 2.0</h1>
<div class=sub>{desc} · {model.num_params():,} params · trained on {n_train} native-2.0(+Space&nbsp;Age) blueprints
(val {n_val} / test {n_test}) · crop {cfg.get('crop_size')} / mask {cfg.get('mask_size')}</div>

<div class=card><b>Held-out test metrics</b> (over masked cells only)
<table><tr><th>metric</th><th>model</th><th>always-EMPTY</th><th>always-majority</th></tr>
{_metric_rows(metrics)}</table>
<div class=lbl>Game renders below are produced by exporting the prediction as a real Factorio 2.0
blueprint string and rendering it with FBSR (the engine FactorioBin uses), so they are pixel-faithful.</div></div>

{cards}
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
