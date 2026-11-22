"""Learned structure model: the teacher's coarse map (7.68 km/px, 6 channels) -> level-0 heights (480 m/px).

It replaces the procedural level 0. It is a small conditional flow-matching model (a few Euler steps), built so that
generation stays stateless and random-access exactly like the students:
  * the network is local only (3x3 convs, no attention, no normalisation layers), so one step has a finite
    receptive field `rf`, and K steps have K * rf;
  * the only randomness is the initial noise, hashed from the absolute level-0 pixel (student_model.hash_normal).
So a window computed with a halo of >= K * rf (+ the bicubic upsampling of the conditioning) and then cropped
depends only on the coarse map, the seed and its absolute position.

Conventions
  * coarse map `c`: (6, h, w) float32 = the teacher's coarse stage output: [elev, p5 elev] in signed-sqrt metres,
    then [mean temp, temp variation, annual precip, precip seasonality]. NaNs (climate over ocean) are allowed.
  * one coarse pixel = cfg["cpp"] level-0 pixels (16: 7.68 km / 480 m) = 256 finest (30 m) pixels.
  * x_0 (level 0, encoded like the students: signed sqrt / norm) = base + res_std * r, where
    base = bicubic_up(c[0]) / norm  and r is what the flow model generates.
  * flow matching: x_t = (1 - t) * r + t * eps, the network predicts v = eps - r; sampling integrates t: 1 -> 0.
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pyramid_data import decode_elev  # noqa: E402
from student_model import hash_normal  # noqa: E402

# normalisation of the teacher's coarse channels (the constants its own latent stage uses for the same channels)
COARSE_MEAN = (14.99, 11.65, 15.87, 619.26, 833.12, 69.40)
COARSE_STD = (21.72, 21.78, 10.40, 452.29, 738.09, 34.59)
N_COND = 6
NOISE_LEVEL_TAG = 1000          # "level" key of hash_normal for the structure model (students use 1..N)
UP_PAD = 2                      # coarse px contaminated at a window border by the bicubic upsampling


def norm_cond(c):
    """(..., 6, h, w) raw coarse channels -> normalised, NaN-free."""
    m = torch.tensor(COARSE_MEAN, device=c.device, dtype=c.dtype).view(-1, 1, 1)
    s = torch.tensor(COARSE_STD, device=c.device, dtype=c.dtype).view(-1, 1, 1)
    return torch.nan_to_num((c - m) / s, nan=0.0, posinf=0.0, neginf=0.0)


def cond_up(c, f):
    """Bicubic upsampling by an integer factor (align_corners=False), (B, C, h, w) -> (B, C, f*h, f*w)."""
    return F.interpolate(c, scale_factor=f, mode="bicubic", align_corners=False)


def make_cond(c, cfg):
    """c: (B, 6, h, w) raw coarse. Returns (cond (B, 6, H, W) for the network, base (B, 1, H, W) encoded heights)."""
    if cfg["space"] != "sqrt":
        raise ValueError("the structure model needs space='sqrt' (the teacher's coarse elevation is signed sqrt)")
    f = cfg["cpp"]
    base = cond_up(torch.nan_to_num(c[:, :1], nan=0.0), f) / cfg["norm"]
    return cond_up(norm_cond(c), f), base


def decode_elev_t(y, norm=50.0):
    """Encoded (signed sqrt / norm) torch tensor -> metres."""
    v = y * norm
    return torch.sign(v) * v * v


class TimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                                         # t: (B,) in [0, 1]
        half = self.dim // 2
        fr = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(1000.0) / (half - 1)))
        a = t.float()[:, None] * 1000.0 * fr[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


class TResBlock(nn.Module):
    """Residual block with a per-sample scale/shift from the time embedding (same value at every pixel: local)."""

    def __init__(self, ch, emb_dim):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(emb_dim, 2 * ch)
        self.act = nn.SiLU()
        with torch.no_grad():
            self.c2.weight.mul_(0.2)
            self.film.weight.zero_()
            self.film.bias.zero_()

    def forward(self, x, e):
        h = self.c1(self.act(x))
        g, b = self.film(e).to(h.dtype).chunk(2, dim=-1)
        h = h * (1 + g[:, :, None, None]) + b[:, :, None, None]
        return x + self.c2(self.act(h))


class StructNet(nn.Module):
    """Time-conditioned local U-Net: forward(x_t, cond, t) -> v. No attention, no normalisation layers."""

    def __init__(self, n_cond=N_COND, base=48, depth=4, blocks=2, emb_dim=128):
        super().__init__()
        self.n_cond, self.depth = n_cond, depth
        w = [base * (2 ** d) for d in range(depth)]
        self.temb = TimeEmb(emb_dim)
        self.stem = nn.Conv2d(1 + n_cond, w[0], 3, padding=1)
        self.enc = nn.ModuleList([nn.ModuleList([TResBlock(w[d], emb_dim) for _ in range(blocks)]) for d in range(depth)])
        self.down = nn.ModuleList([nn.Conv2d(w[d], w[d + 1], 3, stride=2, padding=1) for d in range(depth - 1)])
        self.up = nn.ModuleList([nn.Conv2d(w[d + 1], w[d], 3, padding=1) for d in range(depth - 1)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * w[d], w[d], 1) for d in range(depth - 1)])
        self.dec = nn.ModuleList([nn.ModuleList([TResBlock(w[d], emb_dim) for _ in range(blocks)])
                                  for d in range(depth - 1)])
        self.out = nn.Conv2d(w[0], 1, 3, padding=1)
        with torch.no_grad():
            self.out.weight.mul_(0.01)
            self.out.bias.zero_()

    def forward(self, x, cond, t):
        e = self.temb(t)
        H, W = x.shape[-2:]
        m = 2 ** (self.depth - 1)
        h = torch.cat([x, cond], dim=1)
        ph, pw = (-H) % m, (-W) % m
        if ph or pw:
            h = F.pad(h, (0, pw, 0, ph), mode="replicate")
        h = self.stem(h)
        skips = []
        for d in range(self.depth):
            for blk in self.enc[d]:
                h = blk(h, e)
            if d < self.depth - 1:
                skips.append(h)
                h = self.down[d](h)
        for d in reversed(range(self.depth - 1)):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.fuse[d](torch.cat([self.up[d](h), skips[d]], dim=1))
            for blk in self.dec[d]:
                h = blk(h, e)
        return self.out(F.silu(h))[..., :H, :W]


def time_grid(steps, device=None):
    """t values 1 -> 0 (steps + 1 of them), uniform."""
    return torch.linspace(1.0, 0.0, int(steps) + 1, device=device)


@torch.no_grad()
def sample_flow(model, cond, noise, steps=8, dtype=None):
    """Euler integration of the flow from noise (t=1) to data (t=0). cond/noise: (B, 6/1, H, W). Returns r (B,1,H,W)."""
    x = noise.float()
    ts = time_grid(steps, x.device)
    for k in range(int(steps)):
        t = ts[k].expand(x.shape[0])
        with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=dtype is not None):
            v = model(x, cond, t)
        x = x - (ts[k] - ts[k + 1]) * v.float()
    return x


def effective_rf(model, size=193, thresh=1e-3, ts=(1.0, 0.5, 0.05)):
    """Radius (level-0 px) outside of which an input pixel changes the centre output of ONE step by < thresh x peak
    (max over a few t). The K-step sampler needs a halo of K times this."""
    was = model.training
    model.eval()
    dev = next(model.parameters()).device
    c, r = size // 2, 0
    for tv in ts:
        x = torch.randn(1, 1, size, size, device=dev, requires_grad=True)
        cd = torch.randn(1, model.n_cond, size, size, device=dev, requires_grad=True)
        y = model(x, cd, torch.full((1,), float(tv), device=dev))
        y[0, 0, c, c].backward()
        g = torch.maximum(x.grad.abs().amax(dim=1)[0], cd.grad.abs().amax(dim=1)[0])
        g = g / g.max().clamp_min(1e-30)
        ys, xs = torch.nonzero(g > thresh, as_tuple=True)
        r = max(r, int(max((ys - c).abs().max().item(), (xs - c).abs().max().item())))
        model.zero_grad(set_to_none=True)
    model.train(was)
    return r


def halo_coarse(cfg, steps):
    """Halo (coarse px) a window needs on every side so that its centre is window-independent."""
    return UP_PAD + -(-int(steps) * int(cfg.get("rf", 48)) // cfg["cpp"])


def build_from_cfg(cfg):
    return StructNet(cfg.get("n_cond", N_COND), cfg["base"], cfg["depth"], cfg["blocks"], cfg.get("emb_dim", 128))


def save_structure(path, model, cfg):
    torch.save({"model": model.state_dict(), "cfg": dict(cfg, kind="structure")}, path)


def load_structure(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg = ck["cfg"]
    if cfg.get("kind") != "structure":
        raise ValueError(f"{path} is not a structure-model checkpoint")
    m = build_from_cfg(cfg).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, cfg


@torch.no_grad()
def structure_generate(model, cfg, coarse, coarse_origin=(0, 0), seed=0, steps=None, device=None, dtype=None):
    """coarse: (6, h, w) raw teacher coarse channels; coarse_origin: its top-left in absolute coarse pixels.
    Returns (cpp*h, cpp*w) float32 metres at level 0 (480 m/px). Crop halo_coarse(cfg, steps) coarse px per side."""
    device = torch.device(device) if device is not None else next(model.parameters()).device
    steps = int(steps or cfg.get("steps", 8))
    f = cfg["cpp"]
    c = torch.from_numpy(np.asarray(coarse, np.float32))[None].to(device)
    cond, base = make_cond(c, cfg)
    h, w = base.shape[-2:]
    nz = hash_normal(seed, NOISE_LEVEL_TAG, f * int(coarse_origin[0]), f * int(coarse_origin[1]), h, w, 1)
    r = sample_flow(model, cond, torch.from_numpy(nz)[None].to(device), steps, dtype)
    x = base + cfg["res_std"] * r
    return decode_elev(x[0, 0].cpu().numpy(), cfg["space"], cfg["norm"]).astype(np.float32)
