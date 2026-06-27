"""Decode Factorio blueprint strings into JSON dicts.

A Factorio blueprint string is::

    <version-byte><base64( zlib( utf8-json ) )>

The version byte is currently ``"0"``. We also support a fallback where the
input is already raw JSON (newer tooling sometimes exposes blueprint JSON
directly).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import zlib
from pathlib import Path


class BlueprintDecodeError(ValueError):
    """Raised when a blueprint string cannot be decoded."""


def decode_blueprint_string(s: str) -> dict:
    """Decode a Factorio blueprint string (or raw JSON) into a dict.

    Raises:
        BlueprintDecodeError: on any malformed input.
    """
    if s is None:
        raise BlueprintDecodeError("blueprint string is None")
    if not isinstance(s, (str, bytes)):
        raise BlueprintDecodeError(f"expected str, got {type(s).__name__}")
    if isinstance(s, bytes):
        s = s.decode("utf-8", "replace")

    s = s.strip()
    if not s:
        raise BlueprintDecodeError("empty blueprint string")

    # Fallback: already raw JSON.
    if s[0] in "{[":
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            raise BlueprintDecodeError(f"looks like JSON but failed to parse: {e}") from e
        if not isinstance(obj, dict):
            raise BlueprintDecodeError("decoded JSON is not an object")
        return obj

    # Standard path: strip version byte, base64-decode, zlib-decompress, parse JSON.
    payload = s[1:]
    try:
        raw = base64.b64decode(payload, validate=False)
    except Exception as e:  # binascii.Error and friends
        raise BlueprintDecodeError(f"base64 decode failed: {e}") from e
    if not raw:
        raise BlueprintDecodeError("base64 payload decoded to empty bytes")
    try:
        decompressed = zlib.decompress(raw)
    except zlib.error as e:
        raise BlueprintDecodeError(f"zlib decompress failed: {e}") from e
    try:
        obj = json.loads(decompressed)
    except json.JSONDecodeError as e:
        raise BlueprintDecodeError(f"json parse failed: {e}") from e
    if not isinstance(obj, dict):
        raise BlueprintDecodeError("decoded JSON is not an object")
    return obj


def _iter_blueprint_txt(raw_root: Path):
    """Yield (source_hash, blueprint_txt_path, source_url, info_path) under raw_root."""
    for bp_path in sorted(raw_root.rglob("blueprint.txt")):
        d = bp_path.parent
        source_hash = d.name
        url_path = d / "source_url.txt"
        info_path = d / "info.json"
        source_url = url_path.read_text().strip() if url_path.exists() else None
        yield source_hash, bp_path, source_url, (info_path if info_path.exists() else None)


def decode_raw_dir(raw_root: Path, out_dir: Path) -> dict:
    """Decode every blueprint.txt under raw_root, writing decoded/<hash>.json."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    errors = []
    for source_hash, bp_path, source_url, info_path in _iter_blueprint_txt(raw_root):
        bp_string = bp_path.read_text()
        try:
            data = decode_blueprint_string(bp_string)
        except BlueprintDecodeError as e:
            n_err += 1
            errors.append((source_hash, str(e)))
            print(f"  [error] {source_hash[:12]}: {e}")
            continue
        info = None
        if info_path is not None:
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                info = None
        record = {
            "source_hash": source_hash,
            "source_url": source_url,
            "info": info,
            "data": data,
        }
        (out_dir / f"{source_hash}.json").write_text(json.dumps(record))
        n_ok += 1
        print(f"  [ok]    {source_hash[:12]}  decoded ({len(bp_string)} chars)")
    print(f"\nDecoded {n_ok} blueprint(s), {n_err} error(s) -> {out_dir}")
    return {"ok": n_ok, "errors": n_err, "error_list": errors}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Decode Factorio blueprint strings into JSON.")
    ap.add_argument("--raw", type=Path, required=True, help="raw download root (contains <hash>/blueprint.txt)")
    ap.add_argument("--out", type=Path, required=True, help="output dir for decoded JSON")
    args = ap.parse_args(argv)
    if not args.raw.exists():
        print(f"raw dir not found: {args.raw}", file=sys.stderr)
        return 1
    decode_raw_dir(args.raw, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
