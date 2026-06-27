"""Recursively extract individual blueprints from decoded JSON.

Decoded JSON may be a single ``blueprint``, a ``blueprint_book`` (possibly
nested), or a planner (``upgrade_planner`` / ``deconstruction_planner``).
For this POC we keep only real blueprints that contain entities.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFAULT_MAX_ENTITIES = 20_000
PLANNER_KEYS = ("upgrade_planner", "deconstruction_planner")


def extract_blueprints(
    decoded: dict,
    source_hash: str | None = None,
    max_entities: int = DEFAULT_MAX_ENTITIES,
) -> list[dict]:
    """Flatten a decoded blueprint/book into a list of blueprint records.

    Each record::

        {"source_hash", "label", "entities", "tiles", "item",
         "n_entities", "version", "bp_index"}
    """
    results: list[dict] = []
    counter = {"i": 0}

    def add_blueprint(bp: dict, label_path: list[str]) -> None:
        if not isinstance(bp, dict):
            return
        entities = bp.get("entities") or []
        if not isinstance(entities, list) or len(entities) == 0:
            return  # drop empty blueprints
        if len(entities) > max_entities:
            return  # drop oversized blueprints for v0
        label = bp.get("label")
        if not label and label_path:
            label = " / ".join(p for p in label_path if p)
        idx = counter["i"]
        counter["i"] += 1
        results.append(
            {
                "source_hash": source_hash,
                "bp_index": idx,
                "label": label,
                "item": bp.get("item", "blueprint"),
                "version": bp.get("version"),
                "n_entities": len(entities),
                "entities": entities,
                "tiles": bp.get("tiles"),
            }
        )

    def recurse(node, label_path: list[str]) -> None:
        if not isinstance(node, dict):
            return
        if "blueprint" in node and isinstance(node["blueprint"], dict):
            add_blueprint(node["blueprint"], label_path)
            return
        if "blueprint_book" in node and isinstance(node["blueprint_book"], dict):
            book = node["blueprint_book"]
            book_label = book.get("label") or "book"
            children = book.get("blueprints") or []
            for child in children:
                recurse(child, label_path + [book_label])
            return
        if any(k in node for k in PLANNER_KEYS):
            return  # skip planners
        # Bare blueprint dict (no wrapper) that already carries entities.
        if "entities" in node:
            add_blueprint(node, label_path)

    recurse(decoded, [])
    return results


def extract_decoded_dir(decoded_dir: Path, out_path: Path, max_entities: int) -> dict:
    """Read decoded/<hash>.json files and write one blueprint per line to out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(decoded_dir.glob("*.json"))
    n_files = 0
    n_bp = 0
    entity_counts: list[int] = []
    with out_path.open("w") as out:
        for fp in files:
            try:
                record = json.loads(fp.read_text())
            except Exception as e:
                print(f"  [skip] {fp.name}: {e}")
                continue
            n_files += 1
            data = record.get("data", record)
            source_hash = record.get("source_hash") or fp.stem
            bps = extract_blueprints(data, source_hash=source_hash, max_entities=max_entities)
            for bp in bps:
                bp["source_url"] = record.get("source_url")
                bp["id"] = f"{source_hash[:12]}#{bp['bp_index']}"
                out.write(json.dumps(bp) + "\n")
                n_bp += 1
                entity_counts.append(bp["n_entities"])
            print(f"  [ok] {fp.stem[:12]}: {len(bps)} blueprint(s)")

    entity_counts.sort()
    stats = {
        "decoded_files": n_files,
        "blueprints": n_bp,
        "min_entities": entity_counts[0] if entity_counts else 0,
        "max_entities": entity_counts[-1] if entity_counts else 0,
        "median_entities": int(statistics.median(entity_counts)) if entity_counts else 0,
        "total_entities": sum(entity_counts),
    }
    print(f"\nExtracted {n_bp} blueprint(s) from {n_files} file(s) -> {out_path}")
    print(f"  entities/blueprint: min={stats['min_entities']} "
          f"median={stats['median_entities']} max={stats['max_entities']} "
          f"total={stats['total_entities']}")
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extract individual blueprints from decoded JSON.")
    ap.add_argument("--decoded", type=Path, required=True, help="dir of decoded <hash>.json files")
    ap.add_argument("--out", type=Path, required=True, help="output blueprints.jsonl")
    ap.add_argument("--max-entities", type=int, default=DEFAULT_MAX_ENTITIES)
    args = ap.parse_args(argv)
    if not args.decoded.exists():
        print(f"decoded dir not found: {args.decoded}", file=sys.stderr)
        return 1
    extract_decoded_dir(args.decoded, args.out, args.max_entities)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
