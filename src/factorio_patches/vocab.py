"""Token vocabulary for blueprint grids.

A grid cell token is ``"<entity-name>|<direction>"`` (e.g. ``transport-belt|2``).
Three special tokens always lead the vocab::

    EMPTY (0)   empty cell
    MASK  (1)   masked cell (model input only)
    UNK   (2)   out-of-vocab entity
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

EMPTY = "EMPTY"
MASK = "MASK"
UNK = "UNK"
SPECIAL_TOKENS = [EMPTY, MASK, UNK]
EMPTY_ID, MASK_ID, UNK_ID = 0, 1, 2

# First-scope simplification presets (see README / spec).
BELT_ENTITIES = {
    "transport-belt", "fast-transport-belt", "express-transport-belt",
    "underground-belt", "fast-underground-belt", "express-underground-belt",
    "splitter", "fast-splitter", "express-splitter",
}
LOGISTICS_ENTITIES = BELT_ENTITIES | {
    "inserter", "fast-inserter", "stack-inserter", "long-handed-inserter",
    "burner-inserter", "bulk-inserter",
    "small-electric-pole", "medium-electric-pole", "big-electric-pole", "substation",
    "assembling-machine-1", "assembling-machine-2", "assembling-machine-3",
    "stone-furnace", "steel-furnace", "electric-furnace",
    "pipe", "pipe-to-ground",
}
PRESETS = {"none": None, "belt": BELT_ENTITIES, "logistics": LOGISTICS_ENTITIES}


# Entities whose input/output type is a real (non-derivable) structural choice we
# must predict; we fold it into the token name as a ":input"/":output" suffix.
IO_ENTITIES = ("underground-belt", "loader", "linked-belt")


def entity_token(name: str, direction=None, io=None) -> str:
    """Build the grid token for an entity. Missing direction -> 0.

    ``io`` ("input"/"output", for undergrounds/loaders) is folded into the name as
    a ``:io`` suffix so the model predicts it (it is NOT derivable from layout —
    an input underground diving under a crossing belt looks locally identical to an
    output emerging onto one)."""
    base = f"{name}:{io}" if io else name
    try:
        d = 0 if direction is None else int(direction)
    except (TypeError, ValueError):
        d = 0
    return f"{base}|{d}"


def split_token(token: str):
    """Inverse of entity_token. Returns (name, direction|None). Specials -> (token, None).
    ``name`` may carry an io suffix (e.g. ``underground-belt:output``)."""
    if "|" not in token:
        return token, None
    name, d = token.rsplit("|", 1)
    try:
        return name, int(d)
    except ValueError:
        return name, None


def parse_name_io(name: str):
    """Split an identity name into (real_name, io|None)."""
    if ":" in name:
        n, io = name.rsplit(":", 1)
        if io in ("input", "output"):
            return n, io
    return name, None


def io_for(name: str, entity: dict):
    """The input/output type to encode for this entity, or None."""
    if any(k in name for k in IO_ENTITIES):
        t = entity.get("type")
        if t in ("input", "output"):
            return t
    return None


class Vocab:
    def __init__(self, tokens):
        self.itos = list(tokens)
        if self.itos[:3] != SPECIAL_TOKENS:
            raise ValueError("vocab must start with EMPTY, MASK, UNK")
        self.stoi = {t: i for i, t in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    empty_id = EMPTY_ID
    mask_id = MASK_ID
    unk_id = UNK_ID

    def encode(self, token: str) -> int:
        return self.stoi.get(token, UNK_ID)

    def encode_entity(self, name: str, direction=None) -> int:
        return self.encode(entity_token(name, direction))

    def decode(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < len(self.itos) else UNK

    def is_special(self, idx: int) -> bool:
        return idx in (EMPTY_ID, MASK_ID, UNK_ID)

    @classmethod
    def from_counts(cls, counter: Counter, max_vocab: int = 256, allowed=None) -> "Vocab":
        if max_vocab < len(SPECIAL_TOKENS):
            raise ValueError(f"max_vocab must be >= {len(SPECIAL_TOKENS)} (the special tokens)")
        items = []
        for tok, c in counter.most_common():
            name, _ = split_token(tok)
            if allowed is not None and name not in allowed:
                continue
            items.append((tok, c))
        keep = max(0, max_vocab - len(SPECIAL_TOKENS))
        tokens = SPECIAL_TOKENS + [t for t, _ in items[:keep]]
        return cls(tokens)

    def save(self, path, meta=None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"tokens": self.itos, "specials": SPECIAL_TOKENS, "meta": meta or {}}, indent=2))

    @classmethod
    def load(cls, path) -> "Vocab":
        d = json.loads(Path(path).read_text())
        return cls(d["tokens"])


def count_tokens(blueprints_jsonl: Path):
    """Count entity tokens across a blueprints.jsonl. Returns (Counter, n_blueprints)."""
    counter: Counter = Counter()
    n_bp = 0
    with open(blueprints_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            bp = json.loads(line)
            n_bp += 1
            for e in bp.get("entities", []):
                name = e.get("name")
                counter[entity_token(name, e.get("direction"), io_for(name, e))] += 1
    return counter, n_bp


def build_vocab(blueprints_jsonl: Path, out: Path, max_vocab: int, preset: str) -> Vocab:
    allowed = PRESETS.get(preset)
    counter, n_bp = count_tokens(blueprints_jsonl)
    vocab = Vocab.from_counts(counter, max_vocab=max_vocab, allowed=allowed)

    total = sum(counter.values())
    kept = sum(counter[t] for t in vocab.itos if t not in SPECIAL_TOKENS)
    coverage = (kept / total) if total else 0.0
    meta = {
        "max_vocab": max_vocab,
        "preset": preset,
        "n_blueprints": n_bp,
        "n_distinct_tokens_seen": len(counter),
        "n_entity_instances": total,
        "coverage_of_kept_tokens": round(coverage, 4),
        "vocab_size": len(vocab),
    }
    vocab.save(out, meta=meta)

    print(f"Built vocab: {len(vocab)} tokens (incl. {len(SPECIAL_TOKENS)} specials) -> {out}")
    print(f"  blueprints scanned : {n_bp}")
    print(f"  distinct tokens    : {len(counter)}")
    print(f"  entity instances   : {total}")
    print(f"  preset             : {preset}")
    print(f"  coverage (kept)    : {coverage:.1%}")
    print("  most common tokens :")
    shown = 0
    for tok, c in counter.most_common():
        name, _ = split_token(tok)
        if allowed is not None and name not in allowed:
            continue
        print(f"    {tok:32s} {c:>7d}")
        shown += 1
        if shown >= 20:
            break
    return vocab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a token vocabulary from blueprints.jsonl.")
    ap.add_argument("--blueprints", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-vocab", type=int, default=256)
    ap.add_argument("--preset", choices=list(PRESETS), default="none",
                    help="restrict to a family of entities (others -> UNK)")
    args = ap.parse_args(argv)
    build_vocab(args.blueprints, args.out, args.max_vocab, args.preset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
