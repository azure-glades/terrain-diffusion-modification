"""Student network + position-hashed noise for the stateless coarse-to-fine terrain generator.

Part 1 (numpy only): hash_normal() -- Gaussian noise that is a pure function of (seed, level, absolute pixel).
    Any window that covers pixel (i, j) sees the SAME value there, so overlapping windows agree. This is what
    makes inference stateless / random-access. (Training can use ordinary random noise: it is i.i.d. either way.)
Part 2 (needs torch): StudentNet (local-only U-Net, no attention, no global normalisation), PatchDisc,
    checkpoint helpers and student_generate() for inference on arbitrary windows.

Conventions
  * A level-L student maps  base = up2(x_{L-1})  (+ noise channels)  ->  residual band r_L,  x_L = base + res_std * net(.)
  * MERGED stages: a student may jump several levels at once, from cfg["src"] to cfg["level"] (default src = level-1).
    Then base = up2 applied (level - src) times and the output is 2^(level-src) times larger than the input.
  * All elevations inside the network are in "encoded" space (signed sqrt / norm), see make_pyramid_data.encode_elev.
  * Coordinates: level-L pixel grid. A coarse window whose top-left is coarse pixel (ci, cj) covers fine pixels (2ci, 2cj)...
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pyramid_data import decode_elev, encode_elev  # noqa: E402

# =============================================================== stages (pure python)


def stage_src(cfg):
    """Input level of a student: cfg["src"] for a merged stage, level-1 for the classic per-level students."""
    return int(cfg.get("src", int(cfg["level"]) - 1))


def stage_chain(gens, start, top):
    """gens: {target level: generator with .cfg}. Returns [(src, level, gen)] connecting level `start` to `top`."""
    by_src = {stage_src(g.cfg): g for g in gens.values()}
    out, cur = [], int(start)
    while cur < top:
        g = by_src.get(cur)
        if g is None or int(g.cfg["level"]) > top:
            have = sorted((stage_src(g.cfg), int(g.cfg["level"])) for g in gens.values())
            raise ValueError(f"no student takes level {cur} as input (need a chain {start} -> {top}); have stages {have}")
        out.append((cur, int(g.cfg["level"]), g))
        cur = int(g.cfg["level"])
    return out


# =============================================================== hashed noise (numpy)


def _splitmix64(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def hash_normal(seed, level, i0, j0, h, w, channels=1):
    """(channels, h, w) float32 standard normals for pixels i0..i0+h-1, j0..j0+w-1 of the level-`level` grid."""
    ii = np.arange(i0, i0 + h, dtype=np.int64).astype(np.uint64)[:, None]
    jj = np.arange(j0, j0 + w, dtype=np.int64).astype(np.uint64)[None, :]
    out = np.empty((channels, h, w), np.float32)
    with np.errstate(over="ignore"):
        for c in range(channels):
            key = _splitmix64(np.uint64((int(seed) * 1000003 + int(level) * 8191 + c * 131 + 17) & 0xFFFFFFFFFFFFFFFF))
            a = _splitmix64(ii * np.uint64(0x9E3779B97F4A7C15) + key)
            h1 = _splitmix64(a + jj * np.uint64(0xC2B2AE3D27D4EB4F))
            h2 = _splitmix64(h1 ^ np.uint64(0xD6E8FEB86659FD93))
            u1 = ((h1 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53
            u2 = ((h2 >> np.uint64(11)).astype(np.float64) + 0.5) / 2.0 ** 53
            out[c] = (np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)).astype(np.float32)
    return out


# =============================================================== torch part

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # numpy-only environments can still use hash_normal
    torch = None

if torch is not None:

    def up2_t(x):
        """Bilinear 2x upsample, identical to make_pyramid_data.up2 (align_corners=False)."""
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def stage_k(cfg):
        """Number of 2x levels a student jumps (1 for the classic per-level students)."""
        return int(cfg["level"]) - stage_src(cfg)

    def up_t(x, k=1):
        """up2_t applied k times: the pyramid's upsampling from level L-k to level L."""
        for _ in range(k):
            x = up2_t(x)
        return x

    def stage_forward(model, cfg, src, generator=None, dtype=None):
        """Torch-side stage on a batch of ENCODED maps: src (B, 1, h, w) at level cfg['src'] -> (fine, base), both
        (B, 1, 2^k h, 2^k w) encoded. Noise channels are ordinary random noise (training / chained training input)."""
        base = up_t(src, stage_k(cfg))
        x = base
        if cfg["n_noise"]:
            nz = torch.randn(base.shape[0], cfg["n_noise"], *base.shape[-2:], device=base.device, generator=generator)
            x = torch.cat([base, nz if cfg.get("noise_on", True) else torch.zeros_like(nz)], 1)
        with torch.autocast(device_type=base.device.type, dtype=dtype, enabled=dtype is not None):
            res = model(x, cfg["level"]) if cfg.get("shared") else model(x)
        return base + res.float() * cfg["res_std"], base

    def decode_t(y, space="sqrt", norm=50.0):
        v = y * norm
        return torch.sign(v) * v * v if space == "sqrt" else v

    class ResBlock(nn.Module):
        def __init__(self, ch, emb_dim=0):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1)
            self.act = nn.SiLU()
            with torch.no_grad():
                self.c2.weight.mul_(0.2)      # start near identity without making gradients vanish
            self.film = None
            if emb_dim:                        # per-level scale/shift (same value at every pixel: stays local + stateless)
                self.film = nn.Linear(emb_dim, 2 * ch)
                with torch.no_grad():
                    self.film.weight.zero_()
                    self.film.bias.zero_()

        def forward(self, x, e=None):
            h = self.c1(self.act(x))
            if self.film is not None and e is not None:
                g, b = self.film(e).chunk(2, dim=-1)
                h = h * (1 + g.view(1, -1, 1, 1)) + b.view(1, -1, 1, 1)
            return x + self.c2(self.act(h))

    class CondSeq(nn.Sequential):
        """nn.Sequential of ResBlocks that threads the level embedding through (state-dict keys unchanged)."""

        def forward(self, x, e=None):
            for m in self:
                x = m(x, e)
            return x

    class StudentNet(nn.Module):
        """Local-only U-Net: only 3x3 convs, strided convs and nearest upsampling. No attention, no
        normalisation layers (they would make a pixel depend on the window size and break statelessness).
        n_levels > 0 makes it a SHARED model: forward(x, level) is conditioned on the pyramid level via FiLM."""

        def __init__(self, n_noise=0, base=24, depth=3, blocks=2, n_levels=0, emb_dim=32):
            super().__init__()
            self.n_noise, self.depth, self.n_levels = n_noise, depth, n_levels
            ed = emb_dim if n_levels else 0
            self.emb = nn.Embedding(n_levels, emb_dim) if n_levels else None
            w = [base * (2 ** d) for d in range(depth)]
            self.stem = nn.Conv2d(1 + n_noise, w[0], 3, padding=1)
            self.enc = nn.ModuleList([CondSeq(*[ResBlock(w[d], ed) for _ in range(blocks)]) for d in range(depth)])
            self.down = nn.ModuleList([nn.Conv2d(w[d], w[d + 1], 3, stride=2, padding=1) for d in range(depth - 1)])
            self.up = nn.ModuleList([nn.Conv2d(w[d + 1], w[d], 3, padding=1) for d in range(depth - 1)])
            self.fuse = nn.ModuleList([nn.Conv2d(2 * w[d], w[d], 1) for d in range(depth - 1)])
            self.dec = nn.ModuleList([CondSeq(*[ResBlock(w[d], ed) for _ in range(blocks)]) for d in range(depth - 1)])
            self.out = nn.Conv2d(w[0], 1, 3, padding=1)
            with torch.no_grad():
                self.out.weight.mul_(0.01)
                self.out.bias.zero_()

        def forward(self, x, level=None):
            e = None
            if self.emb is not None:
                if level is None:
                    raise ValueError("shared model: pass level=")
                e = self.emb(torch.as_tensor(int(level), device=x.device))
            H, W = x.shape[-2:]
            m = 2 ** (self.depth - 1)
            ph, pw = (-H) % m, (-W) % m
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            h = self.stem(x)
            skips = []
            for d in range(self.depth):
                h = self.enc[d](h, e)
                if d < self.depth - 1:
                    skips.append(h)
                    h = self.down[d](h)
            for d in reversed(range(self.depth - 1)):
                h = F.interpolate(h, scale_factor=2, mode="nearest")
                h = self.up[d](h)
                h = self.fuse[d](torch.cat([h, skips[d]], dim=1))
                h = self.dec[d](h, e)
            return self.out(F.silu(h))[..., :H, :W]

    class PatchDisc(nn.Module):
        """Local patch discriminator on (base, normalised residual band). No normalisation layers."""

        def __init__(self, ch=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(2, ch, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.LeakyReLU(0.2),
                nn.Conv2d(ch * 4, 1, 3, 1, 1))

        def forward(self, base, res):
            return self.net(torch.cat([base, res], 1))

    def effective_rf(model, size=161, thresh=1e-3, level=None):
        """Radius (in input pixels) outside of which an input pixel changes the centre output by < thresh x peak.
        Measured on the current weights; use the inference halo >= this."""
        was = model.training
        model.eval()
        dev = next(model.parameters()).device
        x = torch.randn(1, 1 + model.n_noise, size, size, device=dev, requires_grad=True)
        y = model(x, level) if getattr(model, "n_levels", 0) else model(x)
        c = size // 2
        y[0, 0, c, c].backward()
        g = x.grad.abs().amax(dim=1)[0]
        g = g / g.max().clamp_min(1e-30)
        ys, xs = torch.nonzero(g > thresh, as_tuple=True)
        r = int(max((ys - c).abs().max().item(), (xs - c).abs().max().item()))
        model.zero_grad(set_to_none=True)
        model.train(was)
        return r

    def save_student(path, model, cfg):
        torch.save({"model": model.state_dict(), "cfg": cfg}, path)

    def build_from_cfg(cfg):
        return StudentNet(cfg["n_noise"], cfg["base"], cfg["depth"], cfg["blocks"], cfg.get("n_levels", 0),
                          cfg.get("emb_dim", 32))

    def load_student(path, device="cpu"):
        ck = torch.load(path, map_location=device)
        cfg = ck["cfg"]
        m = build_from_cfg(cfg).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        return m, cfg

    def level_cfgs(cfg):
        """Per-level configs of a checkpoint: [cfg] for a dedicated student, one entry per trained level for a shared one."""
        if not cfg.get("shared"):
            return [cfg]
        out = []
        for L in cfg["train_levels"]:
            c = dict(cfg, level=L, res_std=cfg["res_std_by_level"][L], rf=cfg["rf_by_level"][L],
                     noise_on=L in cfg["noise_levels"])
            out.append(c)
        return out

    @torch.no_grad()
    def student_generate(model, cfg, coarse_m, coarse_origin=(0, 0), seed=0, device=None, dtype=None):
        """coarse_m: (h, w) metres at the source level (L-1, or cfg["src"] for a merged stage); coarse_origin: its
        top-left in source-level pixels (absolute). Returns (f*h, f*w) metres at level L, f = 2^(L - src).
        Stateless: depends only on the window content, seed and absolute origin."""
        device = torch.device(device) if device is not None else next(model.parameters()).device
        h, w = coarse_m.shape
        c = encode_elev(np.asarray(coarse_m, np.float32), cfg["space"], cfg["norm"]).astype(np.float32)
        f = 2 ** stage_k(cfg)
        base = up_t(torch.from_numpy(c)[None, None].to(device), stage_k(cfg))
        x = base
        if cfg["n_noise"]:
            if cfg.get("noise_on", True):
                nz = hash_normal(seed, cfg["level"], f * coarse_origin[0], f * coarse_origin[1], f * h, f * w,
                                 cfg["n_noise"])
                x = torch.cat([base, torch.from_numpy(nz)[None].to(device)], 1)
            else:                                   # shared model, deterministic level: noise channels held at zero
                x = torch.cat([base, torch.zeros(1, cfg["n_noise"], f * h, f * w, device=device)], 1)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            res = model(x, cfg["level"]) if cfg.get("shared") else model(x)
        fine = base + res.float() * cfg["res_std"]
        return decode_elev(fine[0, 0].cpu().numpy(), cfg["space"], cfg["norm"]).astype(np.float32)