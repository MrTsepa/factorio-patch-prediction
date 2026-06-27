#!/usr/bin/env python
"""Thin wrapper around factorio_patches.train with small POC defaults."""

from __future__ import annotations

import sys

from factorio_patches.train import main

if __name__ == "__main__":
    # Pass through any CLI args; defaults live in factorio_patches.train.
    raise SystemExit(main(sys.argv[1:]))
