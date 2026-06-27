#!/usr/bin/env python
"""End-to-end data prep: decode -> extract -> vocab -> dataset.pt.

Assumes blueprints have already been downloaded into --raw (see
factorio_patches.download_factoriobin). Example:

    python scripts/make_tiny_dataset.py \
        --raw data/raw/factoriobin --processed data/processed \
        --preset logistics --max-vocab 128 --crop-size 64 --mask-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from factorio_patches.blueprint_decode import decode_raw_dir
from factorio_patches.blueprint_extract import extract_decoded_dir
from factorio_patches.dataset import build_dataset
from factorio_patches.rasterize import compute_stats, print_stats
from factorio_patches.vocab import Vocab, build_vocab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/raw/factoriobin"))
    ap.add_argument("--decoded", type=Path, default=Path("data/decoded"))
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--preset", default="logistics", choices=["none", "belt", "logistics"])
    ap.add_argument("--max-vocab", type=int, default=128)
    ap.add_argument("--max-entities", type=int, default=20000)
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--mask-size", type=int, default=16)
    ap.add_argument("--max-dim", type=int, default=256)
    ap.add_argument("--min-entities", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bp_jsonl = args.processed / "blueprints.jsonl"
    vocab_json = args.processed / "vocab.json"
    dataset_pt = args.processed / "dataset.pt"
    args.processed.mkdir(parents=True, exist_ok=True)

    print("== decode ==")
    decode_raw_dir(args.raw, args.decoded)
    print("\n== extract ==")
    extract_decoded_dir(args.decoded, bp_jsonl, args.max_entities)
    print("\n== vocab ==")
    vocab = build_vocab(bp_jsonl, vocab_json, args.max_vocab, args.preset)
    print("\n== rasterize stats ==")
    print_stats(compute_stats(bp_jsonl, vocab, max_dim=args.max_dim))
    print("\n== build dataset ==")
    build_dataset(bp_jsonl, vocab, dataset_pt, crop_size=args.crop_size,
                  mask_size=args.mask_size, max_dim=args.max_dim,
                  min_entities=args.min_entities, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
