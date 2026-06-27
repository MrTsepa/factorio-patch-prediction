"""Polite downloader for FactorioBin posts.

Given a file of known FactorioBin URLs (one per line), fetch each post's
``info.json`` and blueprint string and store them on disk, keyed by the SHA256
of the blueprint string so identical strings dedupe automatically.

We do NOT brute-force post ids. Only the URLs in the seed file are fetched.

FactorioBin API (reverse-engineered):
    GET /post/<id>/info.json          -> {post, node:{type, blueprintStringUrl, ...}}
    GET /post/<id>/<node>/info.json   -> info for a child node
    GET /post/<id>/blueprint.txt      -> 302 redirect to the CDN string
A blueprint-book's root string encodes the whole book, so one book post yields
many individual blueprints after extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

import requests

USER_AGENT = "factorio-patch-poc/0.1 (research; tsepa.stas@gmail.com)"
BASE = "https://factoriobin.com"

POST_RE = re.compile(r"/post/([^/\s?#]+)(?:/(\d+))?")
BARE_RE = re.compile(r"^([A-Za-z0-9_.-]+)(?:/(\d+))?$")


def parse_post_url(url: str):
    """Return (post_id, node_or_None) for a FactorioBin URL or bare id, else None."""
    u = (url or "").strip()
    if not u or u.startswith("#"):
        return None
    m = POST_RE.search(u)
    if m:
        return m.group(1), (int(m.group(2)) if m.group(2) else None)
    m = BARE_RE.match(u)
    if m:
        return m.group(1), (int(m.group(2)) if m.group(2) else None)
    return None


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sleep(sleep_min: float, sleep_max: float) -> None:
    time.sleep(random.uniform(sleep_min, sleep_max))


def _get(session, url, *, accept=None, timeout=30, max_retries=4, sleep_max=5.0):
    """GET with polite retry on 429/5xx. Returns Response (200) or None."""
    headers = {"User-Agent": USER_AGENT}
    if accept:
        headers["Accept"] = accept
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        except requests.RequestException as e:
            wait = min(sleep_max * (attempt + 1), 30)
            print(f"  [warn] request error on {url}: {e}; retry in {wait:.1f}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r
        if r.status_code == 404:
            print(f"  [404]  {url}")
            return None
        if r.status_code in (429, 500, 502, 503, 504):
            ra = r.headers.get("Retry-After")
            wait = float(ra) if (ra and ra.isdigit()) else min(sleep_max * (attempt + 2), 30)
            print(f"  [{r.status_code}]  {url}; backing off {wait:.1f}s")
            time.sleep(wait)
            continue
        print(f"  [{r.status_code}]  {url} (giving up)")
        return None
    print(f"  [fail] exhausted retries for {url}")
    return None


def download_one(session, post_id, node, out_root: Path, sleep_min, sleep_max, force) -> str:
    """Download one post (or node). Returns 'ok' | 'cached' | 'skip' | 'fail'."""
    seg = f"{post_id}" if node is None else f"{post_id}/{node}"
    info_url = f"{BASE}/post/{seg}/info.json"
    print(f"- {seg}")

    info_r = _get(session, info_url, accept="application/json", sleep_max=sleep_max)
    if info_r is None:
        return "fail"
    try:
        info = info_r.json()
    except ValueError:
        print(f"  [skip] info.json was not valid JSON for {seg}")
        return "skip"

    node_info = info.get("node") or {}
    bp_url = node_info.get("blueprintStringUrl") or f"{BASE}/post/{seg}/blueprint.txt"

    _sleep(sleep_min, sleep_max)
    bp_r = _get(session, bp_url, sleep_max=sleep_max)
    if bp_r is None:
        return "fail"
    bp_string = bp_r.text.strip()
    if not bp_string:
        print(f"  [skip] empty blueprint string for {seg}")
        return "skip"

    h = sha256_hex(bp_string)
    dest = out_root / h[:2] / h
    # Treat a download as cached only if the blueprint string is actually on disk
    # (a bare directory can exist from an interrupted run). blueprint.txt is
    # written last, so its presence implies the sidecar files are present too.
    if (dest / "blueprint.txt").exists() and not force:
        print(f"  [cached] {h[:12]} ({len(bp_string)} chars)")
        return "cached"

    dest.mkdir(parents=True, exist_ok=True)
    (dest / "source_url.txt").write_text(f"{BASE}/post/{seg}\n")
    (dest / "info.json").write_text(json.dumps(info, indent=2))
    (dest / "blueprint.txt").write_text(bp_string)  # written last (cache sentinel)
    title = (info.get("post") or {}).get("title")
    ntype = node_info.get("type")
    nent = node_info.get("numEntities")
    print(f"  [saved] {h[:12]}  type={ntype} entities={nent} title={title!r} ({len(bp_string)} chars)")
    return "ok"


def download_urls(urls_path: Path, out_root: Path, sleep_min, sleep_max, force) -> dict:
    out_root.mkdir(parents=True, exist_ok=True)
    lines = urls_path.read_text().splitlines()
    parsed = []
    for ln in lines:
        p = parse_post_url(ln)
        if p:
            parsed.append(p)
    print(f"Read {len(parsed)} URL(s) from {urls_path}\n")

    counts = {"ok": 0, "cached": 0, "skip": 0, "fail": 0}
    with requests.Session() as session:
        for i, (post_id, node) in enumerate(parsed):
            status = download_one(session, post_id, node, out_root, sleep_min, sleep_max, force)
            counts[status] = counts.get(status, 0) + 1
            if i < len(parsed) - 1:
                _sleep(sleep_min, sleep_max)

    print(f"\nDone: {counts['ok']} new, {counts['cached']} cached, "
          f"{counts['skip']} skipped, {counts['fail']} failed -> {out_root}")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Politely download FactorioBin blueprints from a URL list.")
    ap.add_argument("--urls", type=Path, required=True, help="file of FactorioBin post URLs, one per line")
    ap.add_argument("--out", type=Path, required=True, help="output root for raw downloads")
    ap.add_argument("--sleep-min", type=float, default=2.0)
    ap.add_argument("--sleep-max", type=float, default=5.0)
    ap.add_argument("--force", action="store_true", help="redownload even if cached")
    args = ap.parse_args(argv)
    if not args.urls.exists():
        print(f"urls file not found: {args.urls}", file=sys.stderr)
        return 1
    download_urls(args.urls, args.out, args.sleep_min, args.sleep_max, args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
