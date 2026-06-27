#!/usr/bin/env python
"""Thin wrapper around factorio_patches.eval (renders prediction demo PNGs)."""

from __future__ import annotations

import sys

from factorio_patches.eval import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
