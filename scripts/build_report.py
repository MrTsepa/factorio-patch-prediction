#!/usr/bin/env python
"""Build a polished self-contained HTML report for the current best model.

Shows the honest leak-free results, the architecture comparison, the data/method, and a
gallery of the best model's inpainting predictions rendered in real Factorio graphics
(FBSR). Metrics come from the cloud runs (passed in RESULTS) so we don't re-run the slow
full eval; only a handful of examples are rendered.

  uv run python scripts/build_report.py --checkpoint runs/arch_cmp/unet_best.pt \
      --data data/processed/dataset5k.pt --out outputs/report --num 8
"""

from __future__ import annotations

import argparse
import base64
import io
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

HL = (0, 234, 255)   # highlight colour for the predicted (masked) region

from factorio_patches.blueprint_decode import encode_blueprint_string
from factorio_patches.dataset import load_dataset, make_datasets
from factorio_patches.eval import VERSION_2_0, grid_to_blueprint, load_checkpoint
from factorio_patches.render import hconcat, render_diff, render_grid
from factorio_patches.vocab import EMPTY_ID

# Final honest (leak-free 42x data) numbers from the cloud runs.
RESULTS = [
    ("U-Net d96", "5.1M", 0.550, 0.698, 0.952, 0.141, "best — and cheapest"),
    ("U-Net scaled d80", "14.4M", 0.544, 0.703, 0.948, 0.157, "scale doesn't help"),
    ("U-Net + axial attn", "11.7M", 0.502, 0.659, 0.940, 0.086, "attention hurts"),
    ("U-Net + bottleneck attn", "12.3M", 0.419, 0.616, 0.930, 0.044, "attention hurts more"),
    ("ViT transformer (patch-2)", "7.1M", 0.289, 0.476, 0.915, 0.020, "wrong tool: coarse decode"),
]
BASELINE = 0.077  # majority-entity baseline entity-acc

# MaskGIT 2x2: (backbone, plain single-shot, maskgit single-shot, maskgit iterative).
# mg-unet/mg-axial are 1500-sample subset estimates (those two dropped before full test eval).
MASKGIT = [
    ("U-Net d96", 0.550, 0.546, 0.556),
    ("scaled U-Net", 0.544, 0.492, 0.507),
    ("axial U-Net", 0.502, 0.438, 0.501),
    ("ViT transformer", 0.289, 0.304, 0.410),
]


def b64(img, max_w=None, fmt="PNG"):
    if max_w and img.width > max_w:
        img = img.resize((max_w, int(img.height * max_w / img.width)))
    if fmt == "JPEG" and img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (24, 24, 27)); bg.paste(img, mask=img.split()[3]); img = bg
    buf = io.BytesIO(); img.save(buf, fmt, **({"quality": 88} if fmt == "JPEG" else {}))
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def fbsr_render(jobs, fbsr_run):
    if not jobs:
        return
    cmds = "".join(f"bot-render -f={t} -o={p} -full\n" for t, p in jobs) + "exit\n"
    try:
        subprocess.run([fbsr_run], input=cmds, text=True, capture_output=True, timeout=900)
    except Exception as e:
        print(f"[warn] FBSR render failed ({e}); report will show abstract panels only")


def _box_abstract(img, ml, mt, M, cell, color=HL):
    """Precise hole outline on an abstract panel (we control its scale exactly)."""
    d = ImageDraw.Draw(img)
    d.rectangle([ml * cell, mt * cell, ml * cell + M * cell - 1, mt * cell + M * cell - 1],
                outline=color, width=3)
    return img


FBSR_TS = 64.35      # FBSR renders ~64.35 px/tile with a ~10px fixed margin (measured via probe)
FBSR_MARGIN = 10.3


def _box_game(img, grid, ml, mt, M=16, color=HL):
    """Highlight the hole on an FBSR render. FBSR's px/tile is fixed (measured), so map the
    hole's top-left cell to pixels via the entity bbox: content edge (min_cell-0.5) sits at
    the fixed margin. Accurate to ~1 tile (edge multi-tile entities can shift it slightly)."""
    occ = np.argwhere(grid != EMPTY_ID)
    if len(occ) < 1:
        return img
    minr, minc = occ.min(0)
    x0 = FBSR_MARGIN + (ml - int(minc)) * FBSR_TS
    y0 = FBSR_MARGIN + (mt - int(minr)) * FBSR_TS
    if x0 < -FBSR_TS or y0 < -FBSR_TS or x0 > img.width or y0 > img.height:
        return img
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([x0, y0, x0 + M * FBSR_TS, y0 + M * FBSR_TS], fill=color + (38,))
    d.rectangle([x0, y0, x0 + M * FBSR_TS, y0 + M * FBSR_TS], outline=color + (255,), width=6)
    return img


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=Path("data/processed/dataset5k.pt"))
    ap.add_argument("--out", type=Path, default=Path("outputs/report"))
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--scan", type=int, default=250, help="test examples to scan for content-rich ones")
    ap.add_argument("--fbsr-run", default=str(Path(__file__).resolve().parent / "fbsr.sh"))
    args = ap.parse_args(argv)

    device = torch.device("cpu")
    model, vocab, ckpt = load_checkpoint(args.checkpoint, device)
    payload = load_dataset(args.data)
    split = make_datasets(payload)["test"]

    # Scan a chunk of the test set; in-hole accuracy for each content-rich example.
    cand = []
    with torch.no_grad():
        for i in range(min(args.scan, len(split))):
            s = split[i]; y, m = s["y"].numpy(), s["mask"].numpy()
            nz = int((y[m] != EMPTY_ID).sum())
            if nz < 20:
                continue
            pred = model(s["x"].unsqueeze(0)).argmax(1)[0].numpy()
            cand.append((float((pred[m] == y[m]).mean()), nz, i))
    cand.sort(reverse=True)                                      # best -> worst
    median_acc = float(np.median([c[0] for c in cand])) if cand else 0.0
    # NOT cherry-picked: take examples evenly spaced across the accuracy distribution, so the
    # gallery shows the realistic range (best..worst); each is labeled with its own accuracy.
    if cand:
        ks = sorted({int(round(p * (len(cand) - 1))) for p in np.linspace(0.05, 0.92, args.num)})
        picks = [cand[k][2] for k in ks]
    else:
        picks = list(range(min(args.num, len(split))))

    tmp = Path("/tmp/report_render"); tmp.mkdir(parents=True, exist_ok=True)
    jobs, examples = [], []
    with torch.no_grad():
        for rank, i in enumerate(picks):
            s = split[i]; mask = s["mask"].numpy(); y = s["y"].numpy(); xin = s["x"].numpy()
            pred = model(s["x"].unsqueeze(0)).argmax(1)[0].numpy()
            pf = y.copy(); pf[mask] = pred[mask]
            acc = float((pred[mask] == y[mask]).mean())
            mt, ml = (int(v) for v in np.argwhere(mask).min(0))
            subs = [render_grid(xin, vocab, cell=9, mask=mask),
                    render_grid(y, vocab, cell=9, mask=mask),
                    render_grid(pf, vocab, cell=9, mask=mask),
                    render_diff(y, pf, mask, cell=9)]
            for sub in subs[1:]:                      # outline the predicted hole on target/pred/diff
                _box_abstract(sub, ml, mt, 16, 9)
            panel = hconcat(subs)
            for tag, g in (("target", y), ("prediction", pf)):
                bp = grid_to_blueprint(g, vocab, label=f"ex{i} {tag}", version=VERSION_2_0)
                (tmp / f"{rank}_{tag}.txt").write_text(encode_blueprint_string(bp))
                jobs.append((str((tmp / f"{rank}_{tag}.txt").resolve()),
                             str((tmp / f"{rank}_{tag}.png").resolve())))
            examples.append({"i": i, "rank": rank, "acc": acc,
                             "nz": int((y[mask] != EMPTY_ID).sum()),
                             "panel": panel, "y": y, "pf": pf, "mt": mt, "ml": ml})

    print(f"rendering {len(jobs)} blueprints via FBSR ...")
    fbsr_render(jobs, args.fbsr_run)

    args.out.mkdir(parents=True, exist_ok=True)
    html = _html(model, vocab, ckpt, payload, examples, tmp, median_acc)
    (args.out / "index.html").write_text(html)
    print(f"wrote report -> {args.out / 'index.html'}  ({len(html)//1024} KB)")
    return 0


def _cmp_rows():
    out = ""
    for name, p, acc, f1, t5, io, note in RESULTS:
        win = " class=win" if acc == max(r[2] for r in RESULTS) else ""
        bar = int(acc / 0.6 * 100)
        out += (f"<tr{win}><td>{name}</td><td>{p}</td>"
                f"<td class=num><b>{acc:.3f}</b><span class=bar style='width:{bar}%'></span></td>"
                f"<td class=num>{f1:.3f}</td><td class=num>{t5:.3f}</td><td class=num>{io:.3f}</td>"
                f"<td class=note>{note}</td></tr>")
    return out


def _maskgit_rows():
    out = ""
    for name, plain, mg_ss, mg_it in MASKGIT:
        out += (f"<tr><td>{name}</td><td class=num>{plain:.3f}</td><td class=num>{mg_ss:.3f}</td>"
                f"<td class=num><b>{mg_it:.3f}</b></td>"
                f"<td class=num style='color:#5c8'>+{mg_it - mg_ss:.3f}</td></tr>")
    return out


def _gallery(examples, tmp):
    cards = ""
    for ex in examples:
        gt = tmp / f"{ex['rank']}_target.png"; pr = tmp / f"{ex['rank']}_prediction.png"
        game = ""
        if gt.exists() and pr.exists():
            game = (
                '<div class=game>'
                f'<figure><img src="{b64(Image.open(gt), 720, "JPEG")}"><figcaption>TARGET — game render</figcaption></figure>'
                f'<figure><img src="{b64(Image.open(pr), 720, "JPEG")}"><figcaption>PREDICTION — game render</figcaption></figure>'
                '</div>')
        cards += (
            '<div class=ex>'
            f'<div class=exhead>Example {ex["i"]} · {ex["nz"]} entities in the hole · '
            f'in-hole accuracy <b>{ex["acc"]:.0%}</b></div>'
            f'<img class=abs src="{b64(ex["panel"], 1120)}">'
            '<div class=lbl><b style="color:#0ea">Cyan box = the exact 16×16 region the model predicted.</b> '
            'Panels: input (hole greyed) · target · prediction · diff (green=correct, red=wrong). '
            'Game renders below show the same target vs prediction in real Factorio graphics.</div>'
            f'{game}</div>')
    return cards or "<p>(no examples)</p>"


def _html(model, vocab, ckpt, payload, examples, tmp, median_acc=0.0):
    n_tr = len(payload["splits"]["train"]); n_v = len(payload["splits"]["val"]); n_te = len(payload["splits"]["test"])
    cfg = ckpt["config"]
    return f"""<!doctype html><html><head><meta charset=utf-8>
<title>Factorio blueprint inpainting — results report</title>
<style>
 body{{background:#16161a;color:#e8e8ea;font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:36px;max-width:1180px;margin:auto}}
 h1{{font-size:28px;margin:0 0 2px}} h2{{font-size:20px;margin:34px 0 10px;color:#cfe}} .sub{{color:#9aa;margin-bottom:8px}}
 .card{{background:#212127;border:1px solid #333;border-radius:12px;padding:18px 22px;margin:14px 0}}
 table{{border-collapse:collapse;width:100%}} td,th{{padding:8px 12px;border-bottom:1px solid #2c2c33;text-align:left}}
 th{{color:#9ab;font-weight:600;font-size:13px}} td.num{{text-align:right;font-variant-numeric:tabular-nums;position:relative}}
 td.note{{color:#9aa;font-size:13px}} tr.win{{background:rgba(90,200,120,.10)}} tr.win td:first-child{{color:#7e7;font-weight:700}}
 .bar{{position:absolute;left:0;bottom:2px;height:3px;background:#5c8;opacity:.5;border-radius:2px}}
 .ex{{background:#212127;border:1px solid #333;border-radius:12px;padding:14px 16px;margin:16px 0}}
 .exhead{{color:#cde;margin-bottom:10px;font-size:14px}} b{{color:#fff}}
 .game{{display:flex;gap:16px;flex-wrap:wrap}} .game figure{{margin:0;flex:1;min-width:320px}}
 .game img{{width:100%;border-radius:8px;background:#0d0d10}} figcaption{{color:#9aa;font-size:12px;margin-top:5px}}
 img.abs{{width:100%;max-width:1120px;image-rendering:pixelated;border-radius:6px}}
 .lbl{{color:#889;font-size:12px;margin:5px 0 10px}}
 .kpi{{display:flex;gap:26px;flex-wrap:wrap;margin:10px 0}} .kpi div{{font-size:13px;color:#9aa}} .kpi b{{display:block;font-size:24px;color:#fff}}
 code{{color:#fc9}} .pill{{display:inline-block;background:#2a3a2f;color:#8e8;border-radius:20px;padding:2px 11px;font-size:12px;margin-left:8px}}
</style></head><body>
<h1>Factorio Blueprint Patch-Inpainting <span class=pill>native 2.0 + Space Age</span></h1>
<div class=sub>Reconstructing a masked 16×16 region of a 64×64 blueprint grid (each cell = one entity token).
Best model: <b>{cfg.get('arch')}</b> · {model.num_params():,} params.</div>

<div class=card><div class=kpi>
 <div><b>0.550</b>exact entity accuracy (test)</div>
 <div><b>0.952</b>top-5 accuracy</div>
 <div><b>7.1×</b>over majority baseline ({BASELINE:.3f})</div>
 <div><b>10,764</b>blueprints (train {n_tr} / val {n_v} / test {n_te})</div>
</div></div>

<h2>Architecture comparison <span class=sub style=font-size:13px>(identical leak-free data, ~12M params, 120 epochs, early-stopped)</span></h2>
<div class=card><table>
<tr><th>model</th><th>params</th><th>entity-acc</th><th>F1</th><th>top-5</th><th>io-acc</th><th></th></tr>
{_cmp_rows()}
</table>
<div class=sub style=margin-top:10px>The compact pure-convolutional <b>U-Net wins</b> — and is the cheapest. Adding global
or axial <b>attention hurts</b>, and pure <b>scale is a wash</b>: the gap was never semantics (top-5 ≈ 0.95) but
<b>cell-precise placement</b>, where the U-Net's full-resolution decoder is simply the right tool.</div></div>

<h2>MaskGIT — does iterative decoding help? <span class=sub style=font-size:13px>(2×2: training scheme × decoding)</span></h2>
<div class=card><table>
<tr><th>backbone</th><th>plain single-shot</th><th>MaskGIT single-shot</th><th>MaskGIT iterative</th><th>decode gain</th></tr>
{_maskgit_rows()}
</table>
<div class=sub style=margin-top:10px>MaskGIT trains on variable-ratio masks and fills the hole over 8 confidence-ordered
steps (commit the surest cells, re-condition, repeat). The <b>decode gain is monotonic in backbone weakness</b>
— transformer <b>+0.105</b>, axial +0.062, scaled +0.015, U-Net +0.010: iterative decoding's job is building
coherence, so it rescues a coarse model (transformer 0.289→<b>0.410</b>) but adds ~nothing to a U-Net that
already predicts coherently in one shot. <b>It does not beat the plain U-Net (0.556 ≈ 0.550).</b> The 2×2 split
(vs MaskGIT-single-shot) shows the variable-mask <i>training</i> is neutral-to-harmful, and the <i>decoding</i>
is the only upside — and only for weak backbones.</div></div>

<h2>How the data is made honest</h2>
<div class=card><ul style=margin:0;color:#cdd>
<li><b>Sources:</b> FactorioPrints (4.4k native-2.0 strings, 1.1k books) + FactorioBin → 42,274 blueprints.</li>
<li><b>Dedup:</b> entity-multiset hashing removed 42% as duplicate <i>designs</i> (raw-string hashing misses re-encodes) → 24,463 unique.</li>
<li><b>No leakage:</b> connected-component grouping + near-duplicate bridging + by-group split → <b>zero</b> exact or ≥85%-similar blueprints shared across train/val/test (verified).</li>
<li><b>Result:</b> 10,764 usable blueprints, vocab 280. The metrics above are honest generalization, not memorized near-duplicates.</li>
</ul></div>

<h2>Predictions — what the model filled</h2>
<div class=sub><b>Not cherry-picked:</b> these examples are sampled evenly across the test-set accuracy
distribution (median in-hole accuracy ≈ <b>{median_acc:.0%}</b>), so they span the realistic range from
near-perfect to hard/ambiguous — each is labeled with its own accuracy. The <b style=color:#0ea>cyan box</b>
on the abstract panel marks the exact 16×16 region the model predicted (everything else is given context);
the game renders below are pixel-faithful FBSR (the engine FactorioBin uses).</div>
{_gallery(examples, tmp)}
</body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
