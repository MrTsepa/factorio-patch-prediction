"""Factorio blueprint patch-inpainting POC.

Pipeline:
    download_factoriobin -> blueprint_decode -> blueprint_extract
    -> vocab / rasterize -> dataset -> model / train -> eval / render
"""

__version__ = "0.1.0"
