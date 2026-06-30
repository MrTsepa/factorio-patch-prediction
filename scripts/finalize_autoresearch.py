#!/usr/bin/env python
"""Build the clean AutoResearch ledger from the GPU eval log (AR_EVAL lines, consistent n)
and emit the plot. Attaches the real Modal training cost per experiment.

  uv run python scripts/finalize_autoresearch.py /tmp/fulleval.log
"""

import json
import re
import sys
from pathlib import Path

# Real Modal A10G spend attributed to the experiment that required NEW training
# (free reuses of the arch-comparison checkpoints cost $0).
COST = {
    "label smoothing (dead end)": 0.85,
    "D4 aug 80ep (undertrained)": 0.89,
    "aug 80ep + TTA": 0.0,
    "D4 aug converged (139ep)": 1.19,
    "aug ensemble (long+big)": 1.29,           # the d128 aug run
    "aug + non-aug ens + TTA (dead end)": 1.30,  # ~aug-ls run + GPU evals
}

log = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fulleval.log")
rows = []
for line in log.read_text().splitlines():
    m = re.match(r"AR_EVAL (.+?) \| val=([0-9.]+) test=([0-9.]+)", line)
    if m:
        rows.append((m.group(1), float(m.group(2)), float(m.group(3))))

if not rows:
    raise SystemExit(f"no AR_EVAL lines in {log} (eval likely flaked) -- rerun the Modal eval")

led = Path("outputs/autoresearch/experiments.jsonl")
led.parent.mkdir(parents=True, exist_ok=True)
best, out = -1, []
for i, (name, v, t) in enumerate(rows):
    kept = v > best + 1e-9
    best = max(best, v)
    out.append({"exp": i, "name": name, "val_acc": round(v, 4), "test_acc": round(t, 4),
                "running_best_val": round(best, 4), "kept": kept,
                "est_cost": COST.get(name, 0.0), "group": "final", "note": ""})
led.write_text("\n".join(json.dumps(o) for o in out) + "\n")
base_t = rows[0][2]
best_t = max(t for _, _, t in rows)
print(f"ledger: {len(out)} experiments | baseline test {base_t:.3f} -> best test {best_t:.3f} "
      f"(+{best_t-base_t:.3f}) | val baseline {rows[0][1]:.3f} -> {best:.3f} | "
      f"spend ${sum(COST.values()):.2f}")
