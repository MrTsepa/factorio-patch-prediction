#!/usr/bin/env python
"""Plot the AutoResearch trajectory (brain2qwerty-style): running-best val entity-accuracy
over successive experiments, with each experiment as a dot, the cumulative-best step
function, the Default baseline, and annotations of the discovered wins.

  uv run python scripts/plot_autoresearch.py --ledger outputs/autoresearch/experiments.jsonl \
      --out docs/renders/autoresearch.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", type=Path, default=Path("outputs/autoresearch/experiments.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("docs/renders/autoresearch.png"))
    ap.add_argument("--metric", default="val_acc", choices=["val_acc", "test_acc"])
    args = ap.parse_args(argv)

    rows = [json.loads(l) for l in args.ledger.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["exp"])
    xs = [r["exp"] for r in rows]
    ys = [r[args.metric] for r in rows if r.get(args.metric) is not None]
    base = rows[0][args.metric]
    total_cost = sum(r.get("est_cost", 0) for r in rows)

    # running best + the experiments that set a new best (the wins to annotate)
    best, run_best, wins = -1, [], []
    for r in rows:
        v = r.get(args.metric)
        if v is not None and v > best + 1e-9:
            best = v
            wins.append(r)
        run_best.append(best)

    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.axhline(base, ls="--", color="#888", lw=1.6, label="Default (baseline)", zorder=1)
    ax.scatter(xs, [r.get(args.metric) for r in rows], s=44, c="#9bbdd4", alpha=0.7,
               edgecolors="none", zorder=2, label="experiments (incl. dead-ends)")
    ax.step(xs, run_best, where="post", color="#1f6fb2", lw=2.6, zorder=3, label="AutoResearch (running best)")

    # headroom so win labels sit clear of the title
    lo = min(r.get(args.metric) for r in rows) - 0.006
    hi = max(run_best) + 0.026
    ax.set_ylim(lo, hi)

    # annotate wins up-and-left into open space (they climb left-to-right)
    for i, w in enumerate([w for w in wins if w["exp"] != 0]):
        v = w[args.metric]
        ax.scatter([w["exp"]], [v], s=72, color="#1f6fb2", zorder=4)
        ax.annotate(f"+{w['name'].lstrip('+ ').split(' (')[0]}", (w["exp"], v),
                    xytext=(w["exp"] - 1.5, v + 0.011 + 0.003 * (i % 2)),
                    fontsize=10.5, color="#15527f", fontweight="bold", ha="left",
                    arrowprops=dict(arrowstyle="-", color="#15527f", lw=1.1))

    final = run_best[-1]
    base_t, best_t = rows[0]["test_acc"], wins[-1]["test_acc"]
    ax.set_xlabel("Experiment #", fontsize=13)
    ax.set_ylabel("Held-out (val) entity-token accuracy", fontsize=13)
    ax.set_title(f"AutoResearch: Factorio patch-inpainting accuracy\n"
                 f"val {base:.3f} → {final:.3f} (+{final-base:.3f})  ·  "
                 f"test {base_t:.3f} → {best_t:.3f} (+{best_t-base_t:.3f})  ·  "
                 f"{len(rows)} experiments · ~${total_cost:.2f} GPU", fontsize=12, pad=14)
    ax.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.margins(x=0.03)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}  (baseline {base:.3f} -> best {final:.3f}, +{final-base:.3f}; ${total_cost:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
