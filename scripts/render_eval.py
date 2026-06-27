#!/usr/bin/env python
"""Rendering validation harness.

Benchmarks our blueprint renderer (FBSR, the same engine FactorioBin uses) against
FactorioBin's own server-rendered reference images, with quantitative scores and
error-region diagnostics.

Pipeline:
  sample  : pull FactorioBin nodes that have a reference render -> (string, ref.jpg)
  render  : render each string with FBSR (needs the bot-run service alive)
  score   : crop-to-content, align (phase correlation), SSIM + pixel-match%,
            and locate error regions where our render disagrees with the reference

Usage:
  # one-time: build+bake FBSR (see docs/findings.md) and start the service:
  #   scripts/fbsr.sh-service ...   (or: ( echo 'bot-run vanilla -r'; tail -f /dev/null ) | <fbsr-runner> & )
  uv run python scripts/render_eval.py --sample --posts balancers 1o4z16 KafN8H7L --num 6 --out eval_render
  uv run python scripts/render_eval.py --out eval_render          # render (if needed) + score
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image
from skimage.measure import label, regionprops
from skimage.metrics import structural_similarity as ssim
from skimage.registration import phase_cross_correlation

UA = {"User-Agent": "factorio-patch-poc/0.1 (render-eval; tsepa.stas@gmail.com)"}
FBSR_RUN = os.environ.get("FBSR_RUN", str(Path(__file__).resolve().parent / "fbsr.sh"))


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #
def sample_factoriobin(posts, num, out_dir: Path, sleep=2.0):
    out_dir.mkdir(parents=True, exist_ok=True)
    got = 0
    for post in posts:
        per = 0
        for n in range(1, 60):
            if per >= num:
                break
            try:
                r = requests.get(f"https://factoriobin.com/post/{post}/{n}/info.json", headers=UA, timeout=30)
            except requests.RequestException:
                break
            if r.status_code == 404:
                break
            if r.status_code != 200:
                time.sleep(sleep); continue
            nd = r.json().get("node", {})
            time.sleep(sleep)
            if nd.get("type") != "blueprint" or not nd.get("renderImageUrl"):
                continue
            ne = nd.get("numEntities") or 0
            if ne < 6 or ne > 2500:
                continue
            key = f"{post}_{n}"
            txt = out_dir / f"{key}.txt"
            if not txt.exists():
                s = requests.get(nd["blueprintStringUrl"], headers=UA, timeout=60).text.strip()
                txt.write_text(s)
                time.sleep(sleep)
                ext = nd["renderImageUrl"].rsplit(".", 1)[-1]
                img = requests.get(nd["renderImageUrl"], headers=UA, timeout=60).content
                (out_dir / f"{key}_ref.{ext}").write_bytes(img)
                (out_dir / f"{key}.meta.json").write_text(json.dumps(
                    {"post": post, "node": n, "version": nd.get("factorioVersion"),
                     "entities": ne, "name": nd.get("name")}))
                time.sleep(sleep)
            per += 1
            got += 1
            print(f"  sampled {key}  v={nd.get('factorioVersion')} ent={ne} {nd.get('name')!r}")
    print(f"sampled {got} blueprint(s) -> {out_dir}")


# --------------------------------------------------------------------------- #
# render (FBSR)
# --------------------------------------------------------------------------- #
def fbsr_render(jobs, fbsr_run=FBSR_RUN):
    """jobs: list of (txt_path, png_path). One FBSR session renders all of them."""
    if not jobs:
        return
    cmds = "".join(f"bot-render -f={t} -o={p} -full\n" for t, p in jobs) + "exit\n"
    subprocess.run([fbsr_run], input=cmds, text=True, capture_output=True, timeout=600)


# --------------------------------------------------------------------------- #
# align + score
# --------------------------------------------------------------------------- #
def _rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def _bg_color(img):
    s = 10
    corners = np.concatenate([img[:s, :s].reshape(-1, 3), img[:s, -s:].reshape(-1, 3),
                              img[-s:, :s].reshape(-1, 3), img[-s:, -s:].reshape(-1, 3)])
    return np.median(corners, axis=0)


def _content_mask(img, bg, thr=30):
    return np.abs(img.astype(int) - bg.astype(int)).sum(2) > thr


def _crop_to_content(img, pad=3):
    m = _content_mask(img, _bg_color(img))
    ys, xs = np.where(m)
    if len(ys) == 0:
        return img
    r0, r1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad)
    c0, c1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad)
    return img[r0:r1, c0:c1]


def _resize(img, w, h):
    return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))


def _fit(img, H, W):
    """Center pad/crop img to (H, W)."""
    out = np.zeros((H, W, 3), img.dtype)
    h, w = img.shape[:2]
    y0, x0 = (H - h) // 2, (W - w) // 2
    sy, sx, dy, dx = max(0, -y0), max(0, -x0), max(0, y0), max(0, x0)
    hh, ww = min(h - sy, H - dy), min(w - sx, W - dx)
    out[dy:dy + hh, dx:dx + ww] = img[sy:sy + hh, sx:sx + ww]
    return out


def align_and_score(ref_path, ours_path, work_w=440):
    """Both are FBSR renders of the same blueprint; resolve scale + sub-pixel offset,
    then score with mildly-blurred SSIM (robust to JPEG/sub-pixel) + a pixel-match%."""
    from scipy.ndimage import gaussian_filter
    from scipy.ndimage import shift as nd_shift

    ref = _crop_to_content(_rgb(ref_path))
    ours = _crop_to_content(_rgb(ours_path))
    Rh = max(8, int(ref.shape[0] * work_w / ref.shape[1]))
    R = _resize(ref, work_w, Rh)
    Rg = gaussian_filter(R.mean(2), 1.0)
    Rb = gaussian_filter(R.astype(float), (1, 1, 0))

    best = None
    for s in np.linspace(0.90, 1.10, 21):       # coarse scale search
        w = max(8, int(work_w * s))
        h = max(8, int(ours.shape[0] * w / ours.shape[1]))
        O = _fit(_resize(ours, w, h), Rh, work_w)
        sh, _, _ = phase_cross_correlation(Rg, gaussian_filter(O.mean(2), 1.0), upsample_factor=8)
        Oa = nd_shift(O.astype(float), (sh[0], sh[1], 0), order=1, mode="nearest")
        sc = float(ssim(Rb, gaussian_filter(Oa, (1, 1, 0)), channel_axis=2, data_range=255))
        if best is None or sc > best[0]:
            best = (sc, Oa, float(s), [float(sh[0]), float(sh[1])])

    sc, Oa, s, sh = best
    O = np.clip(Oa, 0, 255).astype(np.uint8)
    diff = np.abs(R.astype(int) - O.astype(int)).mean(2)
    match = float((diff < 28).mean())
    return {"ssim": sc, "pixel_match": match, "scale": s, "shift": sh}, R, O, diff


def error_regions(diff, thr=70, min_area=250, topk=12):
    """Localize substantial disagreements (smooth first so sub-pixel edge ghosting
    along matched sprites doesn't register as 'errors')."""
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(diff, 2.0)
    lbl = label(sm > thr)
    regs = sorted(regionprops(lbl), key=lambda r: -r.area)
    regs = [r for r in regs if r.area >= min_area][:topk]
    err_frac = float((sm > thr).mean())
    return [{"bbox": [int(v) for v in r.bbox], "area": int(r.area)} for r in regs], err_frac


def _heatmap(diff):
    d = np.clip(diff / max(1.0, diff.max()), 0, 1)
    hm = np.zeros((*d.shape, 3), np.uint8)
    hm[..., 0] = (d * 255).astype(np.uint8)         # red = error
    hm[..., 2] = ((1 - d) * 60).astype(np.uint8)
    return hm


def _panel(R, O, diff, regs, key, scores):
    from PIL import ImageDraw, ImageFont
    hm = _heatmap(diff).copy()
    pim = Image.fromarray(hm); dr = ImageDraw.Draw(pim)
    for rg in regs:
        r0, c0, r1, c1 = rg["bbox"]
        dr.rectangle([c0, r0, c1, r1], outline=(255, 255, 0), width=1)
    cols = [Image.fromarray(R), Image.fromarray(O), pim]
    titles = ["FactorioBin REFERENCE", "OUR FBSR", f"DIFF  ssim={scores['ssim']:.3f} match={scores['pixel_match']:.0%}"]
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    w = sum(c.width for c in cols) + 20
    H = max(c.height for c in cols) + 28
    out = Image.new("RGB", (w, H), (20, 20, 22)); d = ImageDraw.Draw(out)
    x = 0
    for c, t in zip(cols, titles):
        out.paste(c, (x, 28)); d.text((x + 4, 6), t, fill=(240, 240, 240), font=font)
        x += c.width + 10
    return out


def evaluate(sample_dir: Path, out_dir: Path, fbsr_run=FBSR_RUN, force_render=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted({Path(p).name[:-4] for p in glob.glob(str(sample_dir / "*.txt"))})
    # render any missing
    jobs = []
    for k in keys:
        png = sample_dir / f"{k}_ours.png"
        if force_render or not png.exists():
            # absolute paths: the FBSR runner cd's into its own dir before running
            jobs.append((str((sample_dir / f"{k}.txt").resolve()), str(png.resolve())))
    if jobs:
        print(f"rendering {len(jobs)} blueprint(s) via FBSR ...")
        fbsr_render(jobs, fbsr_run)

    results = []
    for k in keys:
        refs = glob.glob(str(sample_dir / f"{k}_ref.*"))
        ours = sample_dir / f"{k}_ours.png"
        if not refs or not ours.exists():
            print(f"  [skip] {k}: missing ref or render"); continue
        try:
            scores, R, O, diff = align_and_score(refs[0], ours)
        except Exception as e:
            print(f"  [err]  {k}: {e}"); continue
        regs, err_frac = error_regions(diff)
        meta = {}
        mp = sample_dir / f"{k}.meta.json"
        if mp.exists():
            meta = json.loads(mp.read_text())
        _panel(R, O, diff, regs, k, scores).save(out_dir / f"{k}_panel.png")
        rec = {"key": k, **scores, "n_error_regions": len(regs), "error_area": err_frac,
               "version": meta.get("version"), "entities": meta.get("entities"), "name": meta.get("name")}
        results.append(rec)
        print(f"  {k:16s} ssim={scores['ssim']:.3f} match={scores['pixel_match']:.0%} "
              f"err_area={err_frac:.1%} regions={len(regs):2d} v={meta.get('version')} ent={meta.get('entities')}")

    if results:
        ss = np.array([r["ssim"] for r in results])
        mm = np.array([r["pixel_match"] for r in results])
        agg = {"n": len(results), "mean_ssim": float(ss.mean()), "median_ssim": float(np.median(ss)),
               "mean_pixel_match": float(mm.mean()), "min_ssim": float(ss.min())}
        (out_dir / "report.json").write_text(json.dumps({"aggregate": agg, "samples": results}, indent=2))
        print(f"\n=== AGGREGATE over {agg['n']} samples ===")
        print(f"  mean SSIM       : {agg['mean_ssim']:.3f}  (median {agg['median_ssim']:.3f}, min {agg['min_ssim']:.3f})")
        print(f"  mean pixel-match: {agg['mean_pixel_match']:.1%}")
        worst = sorted(results, key=lambda r: r["ssim"])[:3]
        print("  worst samples   :", ", ".join(f"{r['key']}({r['ssim']:.2f})" for r in worst))
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Validate our FBSR renderer against FactorioBin reference renders.")
    ap.add_argument("--out", type=Path, default=Path("eval_render"))
    ap.add_argument("--sample", action="store_true", help="download reference samples first")
    ap.add_argument("--posts", nargs="*", default=["balancers", "1o4z16", "KafN8H7L"])
    ap.add_argument("--num", type=int, default=6, help="samples per post")
    ap.add_argument("--fbsr-run", default=FBSR_RUN)
    ap.add_argument("--force-render", action="store_true")
    args = ap.parse_args(argv)
    samples = args.out / "samples"
    if args.sample:
        sample_factoriobin(args.posts, args.num, samples)
    evaluate(samples, args.out / "results", fbsr_run=args.fbsr_run, force_render=args.force_render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
