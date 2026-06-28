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


def _valid_heads(dim: int, want: int) -> int:
    for h in (want, 8, 4, 2, 1):
        if dim % h == 0:
            return h
    return 1


class AxialAttention(nn.Module):
    """Row-then-column self-attention over a [B,C,H,W] map (O(H*W*(H+W)) not (HW)^2).

    A single row-attention layer can carry an axis-aligned run (belt/pipe/rail) across an
    entire row of the hole in one step — the matching inductive bias for Factorio layouts —
    while keeping full spatial resolution. Pre-norm + residual; position comes from the
    surrounding position-aware conv features.
    """

    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        h = _valid_heads(dim, heads)
        self.n1, self.n2 = _norm(dim), _norm(dim)
        self.row = nn.MultiheadAttention(dim, h, batch_first=True)
        self.col = nn.MultiheadAttention(dim, h, batch_first=True)

    def forward(self, x):                                  # [B,C,H,W]
        B, C, H, W = x.shape
        r = self.n1(x).permute(0, 2, 3, 1).reshape(B * H, W, C)   # rows: attend along W
        r, _ = self.row(r, r, r, need_weights=False)
        x = x + r.reshape(B, H, W, C).permute(0, 3, 1, 2)
        c = self.n2(x).permute(0, 3, 2, 1).reshape(B * W, H, C)   # cols: attend along H
        c, _ = self.col(c, c, c, need_weights=False)
        x = x + c.reshape(B, W, H, C).permute(0, 3, 2, 1)
        return x


class UNet2D(nn.Module):
    """Configurable U-Net backbone: variable depth/width, optional bottleneck self-attention
    (UNETR-lite) and optional axial-attention stages. Subsumes the scaled / attention-augmented
    / axial variants from one class."""

    def __init__(self, vocab_size: int, d_model: int = 96, base: int = 96, depth: int = 2,
                 bottleneck_attn: int = 0, axial_stages: tuple = (), heads: int = 4,
                 grid: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.inc = ConvBlock(d_model, base)
        chs = [base * (2 ** i) for i in range(depth + 1)]          # e.g. depth2 -> [b,2b,4b]
        self.downs = nn.ModuleList([Down(chs[i], chs[i + 1]) for i in range(depth)])
        self.ups = nn.ModuleList([Up(chs[i + 1], chs[i], chs[i]) for i in reversed(range(depth))])
        self.outc = nn.Conv2d(base, vocab_size, 1)

        bdim, bn = chs[-1], grid // (2 ** depth)
        if bottleneck_attn > 0:
            self.bpos = nn.Parameter(torch.randn(1, bn * bn, bdim) * 0.02)
            layer = nn.TransformerEncoderLayer(
                bdim, _valid_heads(bdim, heads), dim_feedforward=4 * bdim,
                batch_first=True, activation="gelu", norm_first=True, dropout=0.0)
            self.battn = nn.TransformerEncoder(layer, bottleneck_attn)
        else:
            self.battn = None
        self.axial = nn.ModuleDict({
            str(grid // (2 ** k)): AxialAttention(chs[k], heads)
            for k in range(depth + 1) if (grid // (2 ** k)) in axial_stages})

    def _maybe_axial(self, f):
        key = str(f.shape[-1])
        return self.axial[key](f) if key in self.axial else f

    def forward(self, x):                                  # [B,H,W] long
        e = self.embed(x).permute(0, 3, 1, 2)
        feats = [self._maybe_axial(self.inc(e))]
        for d in self.downs:
            feats.append(self._maybe_axial(d(feats[-1])))
        b = feats[-1]
        if self.battn is not None:
            B, C, h, w = b.shape
            s = b.flatten(2).transpose(1, 2) + self.bpos
            b = self.battn(s).transpose(1, 2).reshape(B, C, h, w)
        u = b
        for i, up in enumerate(self.ups):
            u = up(u, feats[-(i + 2)])
        return self.outc(u)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(arch: str, vocab_size: int, d_model: int = 64, depth: int = 6,
                heads: int = 6, patch: int = 2):
    if arch == "unet":
        return PatchInpaintUNet(vocab_size, d_model=d_model)
    if arch == "transformer":
        return PatchInpaintTransformer(vocab_size, d_model=d_model, patch=patch,
                                       depth=depth, heads=heads)
    if arch == "unet-scaled":            # control: deeper + wider, no attention
        return UNet2D(vocab_size, d_model=d_model, base=d_model, depth=3, heads=heads)
    if arch == "unet-attn":              # UNETR-lite: U-Net + bottleneck self-attention
        return UNet2D(vocab_size, d_model=d_model, base=d_model, depth=2,
                      bottleneck_attn=max(2, depth), heads=heads)
    if arch == "unet-axial":             # task-matched: U-Net + axial attention (rows/cols)
        return UNet2D(vocab_size, d_model=d_model, base=d_model, depth=2,
                      axial_stages=(16, 32), heads=heads)
    raise ValueError(f"unknown arch: {arch}")
