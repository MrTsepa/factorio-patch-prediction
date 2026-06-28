"""Bulk-fetch native-2.0 blueprint strings from FactorioPrints (Firebase).

FactorioPrints stores ~17.6k blueprints in a public Firebase RTDB. Push-id keys are
chronological, so the NEWEST keys are overwhelmingly Factorio 2.0 (the newest ~3k are
~97% v2). We fetch the newest N keys in chunked, concurrent key-range reads, keep only
v2 strings (no 1.1 migration needed), and store each by sha256 so re-runs and
cross-source duplicates dedupe for free.

    uv run python -m factorio_patches.download_factorioprints --keys 10000 \
        --out data/raw/fprints20 --workers 8

API (Firebase RTDB REST):
    GET /blueprints.json?shallow=true                       -> {key: true, ...}  (all keys)
    GET /blueprints.json?orderBy="$key"&startAt="a"&endAt="b" -> {key: record, ...}
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DB = "https://facorio-blueprints.firebaseio.com"   # note: project name has the typo
UA = {"User-Agent": "factorio-patch-poc/0.1 (research; tsepa.stas@gmail.com)"}


def major_version(s: str) -> int:
    """Factorio major version of a blueprint string (2 for 2.0, 1 for 1.1, 0 for 0.x)."""
    try:
        raw = base64.b64decode(s[1:] + "=" * (-len(s[1:]) % 4))
        d = json.loads(zlib.decompress(raw))
        bp = d.get("blueprint") or d.get("blueprint_book") or {}
        return bp.get("version", 0) >> 48
    except Exception:
        return -1


def is_book(s: str) -> bool:
    try:
        raw = base64.b64decode(s[1:] + "=" * (-len(s[1:]) % 4))
        return '"blueprint_book"' in zlib.decompress(raw)[:64].decode("utf-8", "ignore")
    except Exception:
        return False


def _get(url, params=None, timeout=120, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            wait = min(2 * (attempt + 1), 20)
            print(f"  [{r.status_code}] {url}; retry in {wait}s")
        except requests.RequestException as e:
            wait = min(2 * (attempt + 1), 20)
            print(f"  [warn] {e}; retry in {wait}s")
        time.sleep(wait)
    return None


def fetch_keys() -> list[str]:
    r = _get(f"{DB}/blueprints.json", params={"shallow": "true"})
    keys = sorted((r.json() or {}).keys()) if r else []
    print(f"FactorioPrints index: {len(keys)} blueprint keys")
    return keys


def fetch_range(start_key: str, end_key: str) -> dict:
    r = _get(f"{DB}/blueprints.json",
             params={"orderBy": '"$key"', "startAt": f'"{start_key}"', "endAt": f'"{end_key}"'})
    return (r.json() or {}) if r else {}


def save_record(out: Path, key: str, rec: dict) -> str | None:
    """Persist a v2 blueprint string by hash. Returns 'saved'|'v2-dup'|None(non-v2)."""
    s = (rec or {}).get("blueprintString") or ""
    if not s or major_version(s) != 2:
        return None
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    dest = out / h[:2] / h
    if (dest / "blueprint.txt").exists():
        return "v2-dup"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "meta.json").write_text(json.dumps({
        "key": key, "title": rec.get("title"), "favorites": rec.get("favorites"),
        "createdDate": rec.get("createdDate"), "is_book": is_book(s),
        "source": "factorioprints"}))
    (dest / "blueprint.txt").write_text(s)   # written last (cache sentinel)
    return "saved"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bulk-fetch native-2.0 FactorioPrints blueprints.")
    ap.add_argument("--out", type=Path, default=Path("data/raw/fprints20"))
    ap.add_argument("--keys", type=int, default=10000, help="how many NEWEST keys to scan")
    ap.add_argument("--chunk", type=int, default=300, help="keys per range request")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    keys = fetch_keys()
    if not keys:
        print("no keys fetched; aborting")
        return 1
    newest = keys[-args.keys:]
    chunks = [newest[i:i + args.chunk] for i in range(0, len(newest), args.chunk)]
    print(f"scanning newest {len(newest)} keys in {len(chunks)} chunks "
          f"({args.workers} workers)")

    t0 = time.time()
    saved = dup = nonv2 = 0
    books = 0

    def do_chunk(ch):
        data = fetch_range(ch[0], ch[-1])
        s_local = d_local = n_local = b_local = 0
        for k, rec in data.items():
            r = save_record(args.out, k, rec)
            if r == "saved":
                s_local += 1
                if (rec or {}).get("blueprintString") and is_book(rec["blueprintString"]):
                    b_local += 1
            elif r == "v2-dup":
                d_local += 1
            else:
                n_local += 1
        return s_local, d_local, n_local, b_local

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_chunk, ch): i for i, ch in enumerate(chunks)}
        for done, fut in enumerate(as_completed(futs), 1):
            s, d, n, b = fut.result()
            saved += s; dup += d; nonv2 += n; books += b
            if done % 5 == 0 or done == len(chunks):
                print(f"  [{done}/{len(chunks)}] saved={saved} (books~{books}) "
                      f"dup={dup} non-2.0={nonv2}  {time.time()-t0:.0f}s")

    print(f"\nDone: {saved} new native-2.0 strings ({books} books), {dup} dups, "
          f"{nonv2} non-2.0 skipped -> {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
