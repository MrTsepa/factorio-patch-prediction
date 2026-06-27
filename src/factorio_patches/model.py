"""Small U-Net for blueprint patch inpainting.

Token ids -> embedding -> U-Net (2 downs / 2 ups with skips) -> per-cell logits.
Input  : [B, H, W]      (long token ids)
Output : [B, V, H, W]   (logits over the vocab)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _norm(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), _norm(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(cin, cout)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cin, 2, stride=2)
        self.block = ConvBlock(cin + cskip, cout)

    def forward(self, x, skip):
        x = self.up(x)
        # pad if odd sizes ever occur
        dy, dx = skip.shape[-2] - x.shape[-2], skip.shape[-1] - x.shape[-1]
        if dy or dx:
            x = F.pad(x, [0, dx, 0, dy])
        return self.block(torch.cat([x, skip], dim=1))


class PatchInpaintUNet(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, base: int | None = None):
        super().__init__()
        base = base or d_model
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.inc = ConvBlock(d_model, base)         # H
        self.down1 = Down(base, base * 2)           # H/2
        self.down2 = Down(base * 2, base * 4)        # H/4 (bottleneck)
        self.up1 = Up(base * 4, base * 2, base * 2)  # H/2
        self.up2 = Up(base * 2, base, base)          # H
        self.outc = nn.Conv2d(base, vocab_size, 1)

    def forward(self, x):                      # x: [B, H, W] long
        e = self.embed(x).permute(0, 3, 1, 2)  # [B, d, H, W]
        x1 = self.inc(e)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        u = self.up1(x3, x2)
        u = self.up2(u, x1)
        return self.outc(u)                    # [B, V, H, W]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class PatchInpaintTransformer(nn.Module):
    """ViT-style transformer for patch inpainting.

    Token embedding -> conv 'patchify' stem (64 -> 64/patch) -> learned 2D
    positional embedding -> transformer encoder (global self-attention over the
    patch tokens) -> conv decoder that upsamples back to full resolution, fused
    with a full-res embedding skip so fine local detail survives.
    Input  : [B, H, W] long ;  Output : [B, V, H, W] logits.
    """

    def __init__(self, vocab_size: int, d_model: int = 192, patch: int = 2,
                 depth: int = 6, heads: int = 6, d_embed: int = 64, grid: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch = patch
        n = grid // patch
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.stem = nn.Conv2d(d_embed, d_model, patch, stride=patch)   # grid -> grid/patch
        self.pos = nn.Parameter(torch.randn(1, n * n, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, heads, dim_feedforward=4 * d_model, batch_first=True,
            activation="gelu", norm_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)

        c1, c2 = d_model // 2, d_model // 4
        ups = []
        cin = d_model
        for _ in range(patch.bit_length() - 1):           # log2(patch) upsample x2 blocks
            cout = max(c2, cin // 2)
            ups += [nn.ConvTranspose2d(cin, cout, 2, stride=2),
                    _norm(cout), nn.GELU()]
            cin = cout
        self.up = nn.Sequential(*ups)
        self.fuse = nn.Sequential(nn.Conv2d(cin + d_embed, cin, 3, padding=1),
                                  _norm(cin), nn.GELU())
        self.head = nn.Conv2d(cin, vocab_size, 1)

    def forward(self, x):                      # x: [B, H, W] long
        e = self.embed(x).permute(0, 3, 1, 2)  # [B, d_embed, H, W]
        t = self.stem(e)                       # [B, D, H/p, W/p]
        B, D, h, w = t.shape
        seq = t.flatten(2).transpose(1, 2) + self.pos   # [B, h*w, D]
        seq = self.norm(self.encoder(seq))
        t = seq.transpose(1, 2).reshape(B, D, h, w)
        u = self.up(t)                         # [B, c, H, W]
        u = self.fuse(torch.cat([u, e], dim=1))
        return self.head(u)                    # [B, V, H, W]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arch: str, vocab_size: int, d_model: int = 64, depth: int = 6,
                heads: int = 6, patch: int = 2):
    if arch == "unet":
        return PatchInpaintUNet(vocab_size, d_model=d_model)
    if arch == "transformer":
        return PatchInpaintTransformer(vocab_size, d_model=d_model, patch=patch,
                                       depth=depth, heads=heads)
    raise ValueError(f"unknown arch: {arch}")
