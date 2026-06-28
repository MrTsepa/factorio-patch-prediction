#!/usr/bin/env python
"""One-shot scaled-corpus build: decode raw -> extract+dedup+group -> vocab -> dataset.pt.

Merges multiple raw sources (FactorioBin + FactorioPrints), dedups by entity-multiset
hash (cross-source re-uploads), assigns anti-leakage split groups (connected components
of books sharing any identical blueprint), then builds the vocab + dataset.pt with a
group-aware train/val/test split.

  uv run python scripts/build_corpus.py --out-tag 5k --max-vocab 256
"""

from __future__ import annotations

import argparse
from pathlib import Path

from factorio_patches.blueprint_decode import decode_raw_dir
from factorio_patches.blueprint_extract import extract_corpus
from factorio_patches.dataset import build_dataset
from factorio_patches.vocab import Vocab, count_tokens


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build a scaled, deduped, group-split corpus.")
    ap.add_argument("--raw", nargs="+",
                    default=["data/raw/fbin20:factoriobin",
                             "data/raw/fprints20:factorioprints"],
                    help="raw_dir:source_label entries; LIST ORDER = priority (first wins dups)")
    ap.add_argument("--out-tag", default="5k", help="suffix for blueprints<tag>.jsonl / vocab / dataset")
    ap.add_argument("--max-vocab", type=int, default=256)
    ap.add_argument("--max-dim", type=int, default=384)
    ap.add_argument("--min-dim", type=int, default=24)
    ap.add_argument("--min-entities", type=int, default=8)
    ap.add_argument("--min-invocab-frac", type=float, default=0.5)
    ap.add_argument("--decoded-root", type=Path, default=Path("data/decoded"))
    ap.add_argument("--proc-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--skip-decode", action="store_true", help="reuse existing decoded dirs")
    args = ap.parse_args(argv)

    sources = []
    for prio, spec in enumerate(args.raw):
        raw_dir, _, label = spec.partition(":")
        label = label or Path(raw_dir).name
        dd = args.decoded_root / label
        if args.skip_decode and dd.exists():
            print(f"== reuse decoded {dd} (source={label}) ==")
        else:
            print(f"== decode {raw_dir} (source={label}) -> {dd} ==")
            decode_raw_dir(Path(raw_dir), dd, source=label)
        sources.append((dd, label, prio))

    bp_jsonl = args.proc_root / f"blueprints{args.out_tag}.jsonl"
    print(f"\n== extract + dedup + group -> {bp_jsonl} ==")
    extract_corpus(sources, bp_jsonl, group_mode="component")

    print("\n== vocab ==")
    counter, n_bp = count_tokens(bp_jsonl)
    vocab = Vocab.from_counts(counter, max_vocab=args.max_vocab)
    vocab_path = args.proc_root / f"vocab{args.out_tag}.json"
    vocab.save(vocab_path, meta={"corpus": str(bp_jsonl), "blueprints": n_bp})
    print(f"  vocab {len(vocab)} tokens (from {n_bp} blueprints) -> {vocab_path}")

    ds_path = args.proc_root / f"dataset{args.out_tag}.pt"
    print(f"\n== dataset -> {ds_path} ==")
    build_dataset(bp_jsonl, vocab, ds_path, max_dim=args.max_dim, min_dim=args.min_dim,
                  min_entities=args.min_entities, min_invocab_frac=args.min_invocab_frac)
    print(f"\nDONE -> {ds_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
