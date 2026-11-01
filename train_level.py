#!/usr/bin/env python3
"""Train ONE level of the stateless coarse-to-fine terrain student.

The level-L student learns the residual band  r_L = x_L - up2(x_{L-1})  (what the teacher adds when going from
resolution L-1 to L) from  base = up2(x_{L-1})  plus optional noise channels. Default L = finest level (60 m -> 30 m),
which is the level that costs the teacher the most.

Start deterministic (recommended first run): it tells you how much of the detail is predictable from the coarse map.
  python train_level.py --train data/train --val data/val --out runs/l4 --steps 20000

Then stochastic (noise inputs; add an adversarial loss to keep detail sharp instead of regressing to the mean):
  python train_level.py --train data/train --val data/val --out runs/l4_gan --n-noise 4 --adv-w 0.01 --steps 40000

Evaluate:
  python eval_terrain.py compare --data data/val --level 4 --candidates bilinear,student:runs/l4/best.pt --seam --bench

Notes
  * Everything is local (3x3 convs, no attention/global norm) so any window can be generated with a halo >= the
    measured receptive field and cropped; the script prints that radius.
  * Loss is computed on the residual normalised by its std (res_std, calibrated on the training set).
  * The adversarial path is the least-tested part: get the deterministic run working first.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_model as sm  # noqa: E402
from make_pyramid_data import PyramidData  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--level", type=int, default=None, help="level to train (default: finest = levels-1)")
    ap.add_argument("--space", default="sqrt", choices=["sqrt", "metres"])
    ap.add_argument("--norm", type=float, default=50.0)
    ap.add_argument("--crop", type=int, default=256, help="training crop in fine pixels (multiple of 2^(levels-1))")
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--base", type=int, default=24, help="channels at full resolution (x2 per scale)")
    ap.add_argument("--depth", type=int, default=3, help="number of U-Net scales")
    ap.add_argument("--blocks", type=int, default=2, help="residual blocks per scale")
    ap.add_argument("--n-noise", type=int, default=0, help="noise input channels (0 = deterministic)")
    ap.add_argument("--grad-w", type=float, default=1.0, help="weight of the gradient (finite difference) loss")
    ap.add_argument("--spec-w", type=float, default=0.0,
                    help="weight of the power-spectrum loss on the residual band (0 = off; try 0.05-0.2)")
    ap.add_argument("--slope-w", type=float, default=0.0,
                    help="weight of the slope-distribution loss (0 = off; try 0.1-0.5)")
    ap.add_argument("--tex-start", type=int, default=1000,
                    help="step at which texture losses switch on (ramped in over 1000 steps)")
    ap.add_argument("--adv-w", type=float, default=0.0, help="adversarial loss weight (0 = off)")
    ap.add_argument("--adv-start", type=int, default=2000)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bf16", action="store_true", help="disable bf16 autocast")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def grad_l1(a, b):
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return F.l1_loss(dx(a), dx(b)) + F.l1_loss(dy(a), dy(b))


_SPEC_CACHE = {}


def _spec_bins(h, w, dev, nbins=12, fmin=0.02):
    """Ring index (-1 = unused) of every rfft2 coefficient; rings are log-spaced in cycles/pixel."""
    key = (h, w, str(dev), nbins)
    if key not in _SPEC_CACHE:
        fy = torch.fft.fftfreq(h, device=dev)[:, None]
        fx = torch.fft.rfftfreq(w, device=dev)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        edges = torch.logspace(math.log10(fmin), math.log10(0.5), nbins + 1, device=dev)
        ids = torch.bucketize(r, edges) - 1          # [0, nbins) inside the range
        ids = torch.where((r < fmin) | (r > 0.5), torch.full_like(ids, -1), ids.clamp(max=nbins - 1))
        win = torch.outer(torch.hann_window(h, device=dev), torch.hann_window(w, device=dev))
        _SPEC_CACHE[key] = (ids, win, nbins)
    return _SPEC_CACHE[key]


def radial_log_spectrum(x):
    """Batch-averaged radial log power spectrum of x (B,C,H,W): tensor (nbins,).
    Averaging power over the batch first makes it a statistic of the texture, not of one sample's phases."""
    ids, win, nb = _spec_bins(x.shape[-2], x.shape[-1], x.device)
    X = torch.fft.rfft2(x.float() * win)
    P = (X.real ** 2 + X.imag ** 2).mean(dim=(0, 1))
    out = []
    for k in range(nb):
        m = ids == k
        if m.any():                                  # tiny crops can leave low-frequency rings empty
            out.append(torch.log(P[m].mean() + 1e-8))
    return torch.stack(out)


def spectrum_loss(pred_n, true_n):
    return (radial_log_spectrum(pred_n) - radial_log_spectrum(true_n)).abs().mean()


_QS = None


def slope_quantiles(z_m, nq=32):
    """Quantiles of log1p(|gradient|) of a height field in metres (B,1,H,W); differentiable via sort."""
    global _QS
    dx = z_m[..., :-1, 1:] - z_m[..., :-1, :-1]
    dy = z_m[..., 1:, :-1] - z_m[..., :-1, :-1]
    s = torch.log1p(torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)).flatten()
    s, _ = torch.sort(s)
    if _QS is None or _QS.device != s.device:
        _QS = torch.linspace(0.02, 0.995, nq, device=s.device)
    idx = (_QS * (s.numel() - 1)).long()
    return s[idx]


def slope_loss(pred_m, true_m):
    return (slope_quantiles(pred_m) - slope_quantiles(true_m)).abs().mean()


def forever(loader):
    while True:
        for b in loader:
            yield b


def make_input(base, n_noise, generator=None):
    if n_noise == 0:
        return base
    nz = torch.randn(base.shape[0], n_noise, *base.shape[-2:], device=base.device, generator=generator)
    return torch.cat([base, nz], 1)


@torch.no_grad()
def validate(model, val, L, cfg, dev, n, use_bf16):
    model.eval()
    idx = np.linspace(0, len(val) - 1, min(n, len(val))).astype(int)
    se = ae = bse = bae = 0.0
    for k, i in enumerate(idx):
        item = val[int(i)]
        coarse = torch.from_numpy(item["levels"][L - 1])[None].to(dev)
        fine = torch.from_numpy(item["levels"][L])[None].to(dev)
        base = sm.up2_t(coarse)
        gen = torch.Generator(device=dev).manual_seed(1234 + k)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            r = model(make_input(base, cfg["n_noise"], gen))
        pred = base + r.float() * cfg["res_std"]
        pm, fm, bm = (sm.decode_t(t, cfg["space"], cfg["norm"]) for t in (pred, fine, base))
        ae += (pm - fm).abs().mean().item()
        se += ((pm - fm) ** 2).mean().item()
        bae += (bm - fm).abs().mean().item()
        bse += ((bm - fm) ** 2).mean().item()
    n = len(idx)
    rmse, brmse = math.sqrt(se / n), math.sqrt(bse / n)
    return dict(mae=ae / n, rmse=rmse, base_mae=bae / n, base_rmse=brmse, gain=1.0 - rmse / max(brmse, 1e-9))


def main():
    a = parse()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and not a.no_bf16
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    L = a.level if a.level is not None else a.levels - 1
    if not 1 <= L < a.levels:
        sys.exit(f"--level must be in 1..{a.levels - 1}")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    train = PyramidData(a.train, levels=a.levels, crop=a.crop, space=a.space, norm=a.norm, augment=True, preload=True)
    val = PyramidData(a.val, levels=a.levels, crop=None, space=a.space, norm=a.norm, augment=False, preload=True)
    if len(train) == 0 or len(val) == 0:
        sys.exit(f"empty dataset (train {len(train)}, val {len(val)} patches)")
    print(f"[data] train {len(train)} patches, val {len(val)} patches; training level {L} of {a.levels}")

    state_path = os.path.join(out, "state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False) if (a.resume and os.path.exists(state_path)) else None
    if state is not None:
        cfg = state["cfg"]
    else:
        rng = np.random.default_rng(0)
        res_std = float(np.mean([train[int(i)]["residuals"][L].std() for i in rng.integers(0, len(train), 64)]))
        cfg = dict(level=L, levels=a.levels, n_noise=a.n_noise, base=a.base, depth=a.depth, blocks=a.blocks,
                   space=a.space, norm=a.norm, res_std=res_std)
    print(f"[cfg] {cfg}")

    G = sm.build_from_cfg(cfg).to(dev)
    ema = copy.deepcopy(G).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_par = sum(p.numel() for p in G.parameters())
    rf = sm.effective_rf(G)
    cfg["rf"] = rf
    print(f"[model] {n_par / 1e6:.2f}M parameters; effective receptive field radius ~{rf} fine px "
          f"-> use a halo of >= {rf} fine px ({math.ceil(rf / 2)} coarse px) when generating windows"
          + ("  (WARNING: hit the 80px measurement limit)" if rf >= 79 else ""))

    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    D = optD = None
    if a.adv_w > 0:
        D = sm.PatchDisc().to(dev)
        optD = torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.99))
    start, best = 0, float("inf")
    if state is not None:
        G.load_state_dict(state["G"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        if D is not None and state.get("D") is not None:
            D.load_state_dict(state["D"])
            optD.load_state_dict(state["optD"])
        start, best = state["step"], state["best"]
        print(f"[resume] from step {start}")

    loader = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                         pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    it = forever(loader)
    logf = open(os.path.join(out, "log.jsonl"), "a")
    res_std = cfg["res_std"]
    t0 = time.time()
    acc = dict(l1=0.0, grad=0.0, spec=0.0, slope=0.0, adv=0.0, d=0.0, n=0)

    def lr_at(step):
        warm = min(1.0, (step + 1) / max(a.warmup, 1))
        cos = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / max(a.steps, 1), 1.0)))
        return a.lr * warm * cos

    for step in range(start, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(it)
        coarse = batch["levels"][L - 1].to(dev, non_blocking=True)
        fine = batch["levels"][L].to(dev, non_blocking=True)
        base = sm.up2_t(coarse)
        true_n = (fine - base) / res_std

        G.train()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred_n = G(make_input(base, cfg["n_noise"]))
        pred_n = pred_n.float()
        l1 = F.l1_loss(pred_n, true_n)
        gl = grad_l1(pred_n, true_n)
        loss = l1 + a.grad_w * gl

        use_tex = (a.spec_w > 0 or a.slope_w > 0) and step >= a.tex_start
        if use_tex:
            ramp = min(1.0, (step - a.tex_start + 1) / 1000.0)
            if a.spec_w > 0:
                sl = spectrum_loss(pred_n, true_n)
                loss = loss + ramp * a.spec_w * sl
                acc["spec"] += sl.item()
            if a.slope_w > 0:
                pm = sm.decode_t(base + pred_n * res_std, cfg["space"], cfg["norm"])
                fm = sm.decode_t(fine, cfg["space"], cfg["norm"])
                sp = slope_loss(pm, fm)
                loss = loss + ramp * a.slope_w * sp
                acc["slope"] += sp.item()

        use_adv = D is not None and step >= a.adv_start
        if use_adv:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                d_real = D(base, true_n)
                d_fake = D(base, pred_n.detach())
            d_loss = F.relu(1 - d_real.float()).mean() + F.relu(1 + d_fake.float()).mean()
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
            for p in D.parameters():
                p.requires_grad_(False)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                g_adv = -D(base, pred_n).float().mean()
            for p in D.parameters():
                p.requires_grad_(True)
            loss = loss + a.adv_w * g_adv
            acc["adv"] += g_adv.item()
            acc["d"] += d_loss.item()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - a.ema)
        acc["l1"] += l1.item()
        acc["grad"] += gl.item()
        acc["n"] += 1

        if (step + 1) % a.log_every == 0:
            n = acc["n"]
            sps = (step + 1 - start) / (time.time() - t0)
            print(f"[{step + 1}/{a.steps}] l1 {acc['l1'] / n:.4f} grad {acc['grad'] / n:.4f}"
                  + (f" spec {acc['spec'] / n:.3f}" if use_tex and a.spec_w > 0 else "")
                  + (f" slope {acc['slope'] / n:.3f}" if use_tex and a.slope_w > 0 else "")
                  + (f" adv {acc['adv'] / n:.3f} D {acc['d'] / n:.3f}" if use_adv else "")
                  + f"  lr {lr_at(step):.2e}  {sps:.1f} it/s", flush=True)
            acc = dict(l1=0.0, grad=0.0, spec=0.0, slope=0.0, adv=0.0, d=0.0, n=0)

        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            v = validate(ema, val, L, cfg, dev, a.val_n, use_bf16)
            tag = ""
            if v["mae"] < best:
                best = v["mae"]
                sm.save_student(os.path.join(out, "best.pt"), ema, cfg)
                tag = "  *best*"
            print(f"[val {step + 1}] MAE {v['mae']:.2f} m (bilinear {v['base_mae']:.2f})  "
                  f"RMSE {v['rmse']:.2f} m (bilinear {v['base_rmse']:.2f})  detail gain {100 * v['gain']:.1f}%{tag}",
                  flush=True)
            logf.write(json.dumps(dict(step=step + 1, **v)) + "\n")
            logf.flush()

        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            sm.save_student(os.path.join(out, "last.pt"), ema, cfg)
            torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            D=D.state_dict() if D is not None else None,
                            optD=optD.state_dict() if optD is not None else None,
                            step=step + 1, best=best, cfg=cfg), state_path)

    print(f"[done] best val MAE {best:.2f} m. Checkpoints in {out} (best.pt, last.pt).")


if __name__ == "__main__":
    main()