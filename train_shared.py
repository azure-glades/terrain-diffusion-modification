#!/usr/bin/env python3
"""Train ONE shared student for several pyramid levels (coarse -> fine), conditioned on the level.

The network is the same local-only U-Net as the dedicated students, with a per-level embedding that scales and shifts
the features (FiLM, constant over the image, so statelessness is kept). Each step draws a level, then a batch of
crops sized for that level, and applies that level's loss: L1 + gradient + spectrum + slope, plus a hinge-GAN term on
the levels in --adv-levels (one small discriminator per level, one generator for all).

First experiment (the two stochastic levels only), compare with runs/l1_gan + runs/l2_gan:
  python train_shared.py --train data/train2 --val data/val2 --out runs/shared12 --train-levels 1,2 \
      --noise-levels 1,2 --adv-levels 1,2 --steps 60000 --no-bf16

All four levels in one network (levels 3,4 deterministic, GAN on 1,2):
  python train_shared.py --train data/train2 --val data/val2 --out runs/shared1234 --steps 80000 --no-bf16

Use it:  --student runs/shared12/last.pt,runs/l3/best.pt,runs/l4/best.pt   (any mix of shared and dedicated checkpoints)
Judge GAN levels by the printed hf/rough ratios and by cascade_eval.py / show_levels.py, not by validation gain.
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
from eval_terrain import lap, spectrum_stats  # noqa: E402
from train_level import grad_l1, load_data, slope_loss, spectrum_loss  # noqa: E402

DEF_CROPS = {1: 2048, 2: 1024, 3: 512, 4: 256}      # fine px; chosen so every level trains on ~256 px targets
DEF_BS = {1: 4, 2: 8, 3: 8, 4: 8}


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def kv(s, default, typ=int):
    out = dict(default)
    for part in (s or "").split(","):
        if part.strip():
            k, v = part.split(":")
            out[int(k)] = typ(v)
    return out


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="dir, or several separated by commas")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=5)
    ap.add_argument("--train-levels", default="1,2,3,4")
    ap.add_argument("--noise-levels", default="1,2", help="levels whose input gets noise channels (others get zeros)")
    ap.add_argument("--adv-levels", default="1,2", help="levels trained with the adversarial loss")
    ap.add_argument("--crops", default="", help="per-level crop in finest px, e.g. '1:2048,4:256' (defaults 2048/1024/512/256)")
    ap.add_argument("--bs", default="", help="per-level batch size, e.g. '1:4,4:8'")
    ap.add_argument("--weights", default="", help="per-level sampling weight, e.g. '1:2,2:2,3:1,4:1' (default all 1)")
    ap.add_argument("--space", default="sqrt", choices=["sqrt", "metres"])
    ap.add_argument("--norm", type=float, default=50.0)
    ap.add_argument("--n-noise", type=int, default=2)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--emb-dim", type=int, default=32)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--grad-w", type=float, default=0.2)
    ap.add_argument("--spec-w", type=float, default=0.1)
    ap.add_argument("--slope-w", type=float, default=0.2)
    ap.add_argument("--tex-start", type=int, default=1000)
    ap.add_argument("--adv-w", type=float, default=0.2)
    ap.add_argument("--adv-start", type=int, default=5000)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-n", type=int, default=6, help="validation patches per level")
    ap.add_argument("--val-crop", type=int, default=1024, help="centre crop of each val patch, finest px")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--snapshot-every", type=int, default=5000)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None,
                    help="start from a dedicated student's weights (e.g. runs/l1_gan/last.pt); the new level embeddings "
                         "start at zero, so the shared model initially behaves exactly like that student")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


# ------------------------------------------------------------------ data
def with_crop(ds, crop):
    """Shallow copy of a (possibly concatenated) PyramidData with another crop size; the preloaded arrays are shared."""
    if isinstance(ds, torch.utils.data.ConcatDataset):
        return torch.utils.data.ConcatDataset([with_crop(d, crop) for d in ds.datasets])
    c = copy.copy(ds)
    c.crop = crop
    return c


class LevelDS(torch.utils.data.Dataset):
    def __init__(self, parts):
        self.parts = parts

    def __getitem__(self, key):
        i, L = key
        it = self.parts[L][i]
        return {"coarse": it["levels"][L - 1], "fine": it["levels"][L], "L": L}


class LevelBatches(torch.utils.data.Sampler):
    """Infinite stream of batches; every batch holds crops for ONE level (sizes differ between levels)."""

    def __init__(self, n, levels, probs, bss, seed):
        self.n, self.levels, self.probs, self.bss, self.seed = n, levels, probs, bss, seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        while True:
            L = int(rng.choice(self.levels, p=self.probs))
            yield [(int(i), L) for i in rng.integers(0, self.n, size=self.bss[L])]


# ------------------------------------------------------------------ validation
@torch.no_grad()
def validate(model, val, levels, cfg, dev, n, val_crop, use_bf16, noise_levels):
    model.eval()
    res = {}
    idx = np.linspace(0, len(val) - 1, min(n, len(val))).astype(int)
    for L in levels:
        sf = val_crop // 2 ** (cfg["levels"] - 1 - L)            # target size at level L
        rows = []
        for k, i in enumerate(idx):
            it = val[int(i)]
            c, f = it["levels"][L - 1][0], it["levels"][L][0]
            sc = sf // 2
            ac, bc = (c.shape[0] - sc) // 2, (c.shape[1] - sc) // 2
            c, f = c[ac:ac + sc, bc:bc + sc], f[2 * ac:2 * ac + sf, 2 * bc:2 * bc + sf]
            base = sm.up2_t(torch.from_numpy(np.ascontiguousarray(c))[None, None].to(dev))
            x = base
            if cfg["n_noise"]:
                if L in noise_levels:
                    g = torch.Generator(device=dev).manual_seed(1234 + k)
                    nz = torch.randn(1, cfg["n_noise"], *base.shape[-2:], device=dev, generator=g)
                else:
                    nz = torch.zeros(1, cfg["n_noise"], *base.shape[-2:], device=dev)
                x = torch.cat([base, nz], 1)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                r = model(x, L)
            pred = sm.decode_t(base + r.float() * cfg["res_std_by_level"][L], cfg["space"], cfg["norm"])[0, 0].cpu().numpy()
            true = sm.decode_t(torch.from_numpy(np.ascontiguousarray(f)), cfg["space"], cfg["norm"]).numpy()
            bil = sm.decode_t(base, cfg["space"], cfg["norm"])[0, 0].cpu().numpy()
            rows.append(dict(
                mse=float(((pred - true) ** 2).mean()), bmse=float(((bil - true) ** 2).mean()),
                mae=float(np.abs(pred - true).mean()),
                hf=spectrum_stats(pred)[1] / max(spectrum_stats(true)[1], 1e-12),
                rough=float(lap(pred).std() / max(lap(true).std(), 1e-12))))
        m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        res[L] = dict(mae=m["mae"], rmse=math.sqrt(m["mse"]), gain=1 - math.sqrt(m["mse"] / max(m["bmse"], 1e-12)),
                      hf=m["hf"], rough=m["rough"])
    return res


# ------------------------------------------------------------------ main
def main():
    a = parse()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_bf16 = dev.type == "cuda" and not a.no_bf16
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    levels = ints(a.train_levels)
    noise_levels, adv_levels = set(ints(a.noise_levels)), set(ints(a.adv_levels)) & set(levels)
    crops, bss = kv(a.crops, DEF_CROPS), kv(a.bs, DEF_BS)
    wts = kv(a.weights, {L: 1.0 for L in levels}, float)
    if any(not 1 <= L < a.levels for L in levels):
        sys.exit(f"--train-levels must be within 1..{a.levels - 1}")
    for L in levels:
        if crops[L] % 2 ** (a.levels - 1):
            sys.exit(f"crop for level {L} must be divisible by {2 ** (a.levels - 1)}")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    full = load_data(a.train, levels=a.levels, crop=None, space=a.space, norm=a.norm, augment=True, preload=True)
    val = load_data(a.val, levels=a.levels, crop=None, space=a.space, norm=a.norm, augment=False, preload=True)
    if len(full) == 0 or len(val) == 0:
        sys.exit(f"empty dataset (train {len(full)}, val {len(val)} patches)")
    parts = {L: with_crop(full, crops[L]) for L in levels}
    print(f"[data] train {len(full)} patches, val {len(val)} patches; shared student for levels {levels}; "
          f"noise on {sorted(noise_levels)}, adversarial on {sorted(adv_levels)}")

    state_path = os.path.join(out, "state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False) if (a.resume and os.path.exists(state_path)) else None
    if state is not None:
        cfg = state["cfg"]
    else:
        rng = np.random.default_rng(0)
        res_std = {L: float(np.mean([parts[L][int(i)]["residuals"][L].std()
                                     for i in rng.integers(0, len(full), 32)])) for L in levels}
        cfg = dict(level=None, levels=a.levels, shared=True, n_levels=a.levels, emb_dim=a.emb_dim, n_noise=a.n_noise,
                   base=a.base, depth=a.depth, blocks=a.blocks, space=a.space, norm=a.norm,
                   train_levels=levels, noise_levels=sorted(noise_levels), res_std_by_level=res_std, rf_by_level={})
    print(f"[cfg] {cfg}")

    G = sm.build_from_cfg(cfg).to(dev)
    if a.init and state is None:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        for k in ("base", "depth", "blocks", "n_noise"):
            if ck["cfg"].get(k) != cfg[k]:
                sys.exit(f"--init checkpoint has {k}={ck['cfg'].get(k)} but this run uses {k}={cfg[k]}; pass matching "
                         f"--{k.replace('_', '-')}")
        r = G.load_state_dict(ck["model"], strict=False)
        print(f"[init] loaded {a.init}; new (zero-initialised) tensors: {len(r.missing_keys)}, unexpected: {len(r.unexpected_keys)}")
    ema = copy.deepcopy(G).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[model] {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M parameters (one network for levels {levels})")
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    Ds = {L: sm.PatchDisc().to(dev) for L in sorted(adv_levels)}
    optDs = {L: torch.optim.Adam(D.parameters(), lr=a.lr, betas=(0.5, 0.99)) for L, D in Ds.items()}
    start, best = 0, float("inf")
    if state is not None:
        G.load_state_dict(state["G"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        for L, D in Ds.items():
            if state.get("Ds", {}).get(L) is not None:
                D.load_state_dict(state["Ds"][L])
                optDs[L].load_state_dict(state["optDs"][L])
        start, best = state["step"], state["best"]
        print(f"[resume] from step {start}")

    p = np.array([wts[L] for L in levels], dtype=float)
    sampler = LevelBatches(len(full), levels, p / p.sum(), bss, a.seed + start)
    loader = torch.utils.data.DataLoader(LevelDS(parts), batch_sampler=sampler, num_workers=a.workers,
                                         pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)
    it = iter(loader)
    logf = open(os.path.join(out, "log.jsonl"), "a")
    t0 = time.time()
    keys = ("l1", "grad", "spec", "slope", "adv", "d", "n")
    acc = {L: dict.fromkeys(keys, 0.0) for L in levels}
    best_levels = [L for L in levels if L not in adv_levels]

    def lr_at(step):
        warm = min(1.0, (step + 1) / max(a.warmup, 1))
        return a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / max(a.steps, 1), 1.0))))

    def save_ckpt(path):
        cfg["rf_by_level"] = {L: sm.effective_rf(ema, level=L) for L in levels}
        sm.save_student(path, ema, cfg)

    for step in range(start, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = next(it)
        L = int(batch["L"][0])
        coarse = batch["coarse"].to(dev, non_blocking=True)
        fine = batch["fine"].to(dev, non_blocking=True)
        rs = cfg["res_std_by_level"][L]
        base = sm.up2_t(coarse)
        true_n = (fine - base) / rs
        x = base
        if cfg["n_noise"]:
            nz = torch.randn(base.shape[0], cfg["n_noise"], *base.shape[-2:], device=dev) if L in noise_levels \
                else torch.zeros(base.shape[0], cfg["n_noise"], *base.shape[-2:], device=dev)
            x = torch.cat([base, nz], 1)
        G.train()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
            pred_n = G(x, L)
        pred_n = pred_n.float()
        l1 = F.l1_loss(pred_n, true_n)
        gl = grad_l1(pred_n, true_n)
        loss = l1 + a.grad_w * gl
        ac = acc[L]
        if (a.spec_w > 0 or a.slope_w > 0) and step >= a.tex_start:
            ramp = min(1.0, (step - a.tex_start + 1) / 1000.0)
            if a.spec_w > 0:
                sl = spectrum_loss(pred_n, true_n)
                loss = loss + ramp * a.spec_w * sl
                ac["spec"] += sl.item()
            if a.slope_w > 0:
                pm = sm.decode_t(base + pred_n * rs, cfg["space"], cfg["norm"])
                sp = slope_loss(pm, sm.decode_t(fine, cfg["space"], cfg["norm"]))
                loss = loss + ramp * a.slope_w * sp
                ac["slope"] += sp.item()
        if L in Ds and step >= a.adv_start:
            D, optD = Ds[L], optDs[L]
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                d_real, d_fake = D(base, true_n), D(base, pred_n.detach())
            d_loss = F.relu(1 - d_real.float()).mean() + F.relu(1 + d_fake.float()).mean()
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()
            for q in D.parameters():
                q.requires_grad_(False)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16):
                g_adv = -D(base, pred_n).float().mean()
            for q in D.parameters():
                q.requires_grad_(True)
            loss = loss + a.adv_w * g_adv
            ac["adv"] += g_adv.item()
            ac["d"] += d_loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, q in zip(ema.parameters(), G.parameters()):
                pe.lerp_(q, 1.0 - a.ema)
        ac["l1"] += l1.item()
        ac["grad"] += gl.item()
        ac["n"] += 1

        if (step + 1) % a.log_every == 0:
            sps = (step + 1 - start) / (time.time() - t0)
            parts_txt = []
            for Lk, v in acc.items():
                n = max(v["n"], 1)
                if v["n"]:
                    parts_txt.append(f"L{Lk}[l1 {v['l1'] / n:.3f} gr {v['grad'] / n:.3f}"
                                     + (f" sp {v['spec'] / n:.2f}" if v["spec"] else "")
                                     + (f" adv {v['adv'] / n:.2f} D {v['d'] / n:.2f}" if v["d"] else "") + "]")
            print(f"[{step + 1}/{a.steps}] " + " ".join(parts_txt) + f"  lr {lr_at(step):.1e}  {sps:.1f} it/s", flush=True)
            acc = {Lk: dict.fromkeys(keys, 0.0) for Lk in levels}

        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            v = validate(ema, val, levels, cfg, dev, a.val_n, a.val_crop, use_bf16, noise_levels)
            for Lk, r in v.items():
                print(f"[val {step + 1}] L{Lk}: MAE {r['mae']:.2f} m  RMSE {r['rmse']:.2f} m  gain {100 * r['gain']:.1f}%  "
                      f"hf/teacher {r['hf']:.2f}  rough/teacher {r['rough']:.2f}"
                      + ("   (GAN level: judge by hf/rough and by eye)" if Lk in adv_levels else ""), flush=True)
            logf.write(json.dumps(dict(step=step + 1, val={str(k): r for k, r in v.items()})) + "\n")
            logf.flush()
            if best_levels:
                score = float(np.mean([v[Lk]["mae"] for Lk in best_levels]))
                if score < best:
                    best = score
                    save_ckpt(os.path.join(out, "best.pt"))
                    print(f"[val {step + 1}] new best.pt (mean MAE of non-GAN levels {best_levels}: {best:.2f} m)", flush=True)

        if a.snapshot_every and (step + 1) % a.snapshot_every == 0:
            save_ckpt(os.path.join(out, f"snap_{step + 1:06d}.pt"))
        stop = bool(a.max_minutes) and (time.time() - t0) / 60.0 >= a.max_minutes
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or stop:
            save_ckpt(os.path.join(out, "last.pt"))
            torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(),
                            Ds={L: D.state_dict() for L, D in Ds.items()},
                            optDs={L: o.state_dict() for L, o in optDs.items()},
                            step=step + 1, best=best, cfg=cfg), state_path)
        if stop and step + 1 < a.steps:
            print(f"[time limit] stopped at step {step + 1}/{a.steps}; state saved in {out} -- rerun to resume", flush=True)
            return
    print(f"[done] checkpoints in {out} (last.pt, snap_*.pt" + (", best.pt" if best_levels else "") + ")")


if __name__ == "__main__":
    main()