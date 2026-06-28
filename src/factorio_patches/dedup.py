"""Translation-invariant entity-multiset hashing for cross-source dedup.

Raw-string SHA256 (the raw/<hash> dirs) only catches byte-identical re-uploads; it
misses re-encodes (different JSON key/entity order, version bumps), translated copies,
and label/icon-only edits. FactorioPrints heavily re-uploads FactorioBin content, so we
dedup on a canonical multiset of the *structural identity the model sees*:

  (name [+ :input/:output io], direction, position quantized to half-tiles, shifted
   so min(x)=min(y)=0), sorted.

Equal multisets -> equal sorted tuple -> equal hash. This is translation-invariant but
NOT rotation/mirror-invariant by design (a rotated factory is a different example).
"""

from __future__ import annotations

import hashlib

from .vocab import io_for


def _q(v: float) -> int:
    """Half-tile -> int (Factorio positions are multiples of 0.5; robust to float noise)."""
    return int(round(v * 2.0))


def entity_signature(entities: list[dict]) -> tuple | None:
    rows = []
    for e in entities:
        pos = e.get("position") or {}
        x, y = pos.get("x"), pos.get("y")
        if x is None or y is None:
            continue
        name = e.get("name")
        io = io_for(name, e)                       # 'input'/'output' or None
        ident = f"{name}:{io}" if io else name
        try:
            d = int(e.get("direction") or 0)
        except (TypeError, ValueError):
            d = 0
        rows.append((ident, d, _q(x), _q(y)))
    if not rows:
        return None
    mnx = min(r[2] for r in rows)
    mny = min(r[3] for r in rows)
    rows = [(i, d, rx - mnx, ry - mny) for (i, d, rx, ry) in rows]
    rows.sort()
    return tuple(rows)


def entity_multiset_hash(entities: list[dict]) -> str | None:
    sig = entity_signature(entities)
    if sig is None:
        return None
    return hashlib.blake2b(repr(sig).encode("utf-8"), digest_size=16).hexdigest()
