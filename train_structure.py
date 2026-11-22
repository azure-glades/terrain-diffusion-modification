#!/usr/bin/env python3
"""Train the structure model: teacher coarse map (7.68 km/px) -> level 0 (480 m/px). See structure_model.py.

Data: teacher datasets with 2048 px patches that have coarse sidecars (python make_pyramid_data.py add-coarse ...).
  python train_structure.py --train data/train2,data/train3 --val data/val2,data/val3 --out runs/struct

It is a conditional flow-matching model, so there is no adversarial loss to balance: the training loss is a plain
regression and the validation loss is meaningful (lower = better). Every validation also SAMPLES a few maps with the
few-step sampler and compares their texture statistics with the teacher's (1.00 = match).

Resumable (--resume) and time-limited (--max-minutes) like train_level.py.
"""
import os
import sys

# Always run with the project's virtualenv (.venv next to this file), whatever `python` launched the script.
_VENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
_VENV_PY = os.path.join(_VENV, "bin", "python")
if os.path.exists(_VENV_PY) and os.path.realpath(sys.prefix) != os.path.realpath(_VENV):
    os.execv(_VENV_PY, [_VENV_PY] + sys.argv)

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import structure_model as stm  # noqa: E402
from make_pyramid_data import StructData  # noqa: E402


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True, help="dataset dir(s), comma separated")
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--space", default="sqrt", choices=["sqrt"])
    ap.add_argument("--norm", type=float, default=50.0)
    ap.add_argument("--crop", type=int, default=128, help="training crop in level-0 px (<= patch / 16; multiple of 8)")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--base", type=int, default=48)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--blocks", type=int, default=2)
    ap.add_argument("--sample-steps", type=int, default=8, help="Euler steps of the sampler (stored as the default)")
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-n", type=int, default=32)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--max-minutes", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_data(spec, **kw):
    parts = [StructData(p.strip(), **kw) for p in spec.split(",") if p.strip()]
    for p in parts[1:]:
        if (p.S, p.cpp, p.margin) != (parts[0].S, parts[0].cpp, parts[0].margin):
            sys.exit("all datasets must share the patch size and coarse margin")
    ds = parts[0] if len(parts) == 1 else torch.utils.data.ConcatDataset(parts)
    return ds, parts[0]


def prepare(batch, cfg, margin, dev):
    """batch -> (r0 normalised residual target, cond, base), all (B, C, S, S) on the patch footprint."""
    x = batch["x"].to(dev, non_blocking=True)
    cond, base = stm.make_cond(batch["coarse"].to(dev, non_blocking=True), cfg)
    S, o = x.shape[-1], margin * cfg["cpp"]
    cond, base = cond[..., o:o + S, o:o + S], base[..., o:o + S, o:o + S]
    return (x - base) / cfg["res_std"], cond, base


def band_std(z, k=5):
    """Std of the high-pass (z - box blur), per sample: a cheap 'how much fine texture' statistic."""
    return (z - F.avg_pool2d(F.pad(z, (k // 2,) * 4, mode="replicate"), k, 1)).flatten(1).std(dim=1)


def slope_mean(z):
    return torch.sqrt((z[..., 1:, :-1] - z[..., :-1, :-1]) ** 2 + (z[..., :-1, 1:] - z[..., :-1, :-1]) ** 2).flatten(1).mean(1)


@torch.no_grad()
def validate(model, val, cfg, margin, dev, n, amp, steps):
    model.eval()
    idx = np.linspace(0, len(val) - 1, min(n, len(val))).astype(int)
    loss, hf, sl, k = 0.0, [], [], 0
    for b0 in range(0, len(idx), 8):
        items = [val[int(i)] for i in idx[b0:b0 + 8]]
        batch = {key: torch.from_numpy(np.stack([it[key] for it in items])) for key in ("x", "coarse")}
        r0, cond, base = prepare(batch, cfg, margin, dev)
        gen = torch.Generator(device=dev).manual_seed(4321 + b0)
        eps = torch.randn(r0.shape, device=dev, generator=gen)
        for tv in (0.1, 0.3, 0.5, 0.7, 0.9):                       # fixed t grid and noise: comparable across runs
            t = torch.full((r0.shape[0],), tv, device=dev)
            with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
                v = model((1 - tv) * r0 + tv * eps, cond, t)
            loss += F.mse_loss(v.float(), eps - r0).item() * r0.shape[0] / 5
        if b0 == 0:                                               # sampled texture statistics on the first batch
            r = stm.sample_flow(model, cond, eps, steps, amp)
            dec = lambda q: stm.decode_elev_t(base + cfg["res_std"] * q, cfg["norm"])  # noqa: E731
            pm, tm = dec(r), dec(r0)
            hf = (band_std(pm) / band_std(tm).clamp_min(1e-6)).tolist()
            sl = (slope_mean(pm) / slope_mean(tm).clamp_min(1e-6)).tolist()
        k += r0.shape[0]
    return dict(loss=loss / max(k, 1), hf=float(np.median(hf)), slope=float(np.median(sl)))


def main():
    a = parse()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = torch.bfloat16 if (dev.type == "cuda" and not a.no_bf16) else None
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    train, t0d = load_data(a.train, space=a.space, norm=a.norm, augment=True)
    val, _ = load_data(a.val, space=a.space, norm=a.norm, augment=False)
    if len(train) == 0 or len(val) == 0:
        sys.exit(f"empty dataset (train {len(train)}, val {len(val)} patches); the structure model needs 2048 px patches")
    S, margin = t0d.S, t0d.margin
    crop = min(a.crop, S)
    if crop % 8:
        sys.exit("--crop must be a multiple of 8")
    print(f"[data] train {len(train)} patches, val {len(val)} patches of {S}x{S} level-0 px "
          f"({S * 0.48:.0f} km); crop {crop}")
    if len(train) < 2000:
        print(f"[data] NOTE: {len(train)} patches is little for a generative model; watch val loss for overfitting "
              f"and add teacher data (make_pyramid_data.py generate --patch 2048 --min-std 0 --max-ocean 1) if it rises")

    state_path = os.path.join(out, "state.pt")
    state = torch.load(state_path, map_location="cpu", weights_only=False) if (a.resume and os.path.exists(state_path)) else None
    if state is not None:
        cfg = state["cfg"]
    else:
        cfg = dict(kind="structure", cpp=t0d.cpp, n_cond=stm.N_COND, base=a.base, depth=a.depth, blocks=a.blocks,
                   space=a.space, norm=a.norm, steps=a.sample_steps, mpp=30.0 * 256 / t0d.cpp, res_std=1.0)
        rng = np.random.default_rng(0)
        items = [train[int(i)] for i in rng.integers(0, len(train), min(64, len(train)))]
        batch = {k: torch.from_numpy(np.stack([it[k] for it in items])) for k in ("x", "coarse")}
        r, _, _ = prepare(batch, cfg, margin, torch.device("cpu"))
        cfg["res_std"] = float(r.std())
    print(f"[cfg] {cfg}")

    G = stm.build_from_cfg(cfg).to(dev)
    ema = copy.deepcopy(G).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    print(f"[model] {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M parameters")
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.9, 0.99), weight_decay=0.0)
    start, best = 0, float("inf")
    if state is not None:
        G.load_state_dict(state["G"])
        ema.load_state_dict(state["ema"])
        opt.load_state_dict(state["opt"])
        start, best = state["step"], state["best"]
        print(f"[resume] from step {start}")

    loader = torch.utils.data.DataLoader(train, batch_size=a.bs, shuffle=True, num_workers=a.workers, drop_last=True,
                                         pin_memory=dev.type == "cuda", persistent_workers=a.workers > 0)

    def forever():
        while True:
            yield from loader

    def lr_at(step):
        warm = min(1.0, (step + 1) / max(a.warmup, 1))
        return a.lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step / max(a.steps, 1), 1.0))))

    def save(path, model):
        cfg["rf"] = stm.effective_rf(model)                       # measured on the weights being saved
        stm.save_structure(path, model, cfg)

    it = forever()
    logf = open(os.path.join(out, "log.jsonl"), "a")
    t0, acc, n_acc = time.time(), 0.0, 0
    for step in range(start, a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        r0, cond, _ = prepare(next(it), cfg, margin, dev)
        if crop < S:
            i, j = (int(v) for v in torch.randint(0, S - crop + 1, (2,)))
            r0, cond = r0[..., i:i + crop, j:j + crop], cond[..., i:i + crop, j:j + crop]
        t = torch.rand(r0.shape[0], device=dev)
        eps = torch.randn_like(r0)
        tt = t.view(-1, 1, 1, 1)
        G.train()
        with torch.autocast(device_type=dev.type, dtype=amp, enabled=amp is not None):
            v = G((1 - tt) * r0 + tt * eps, cond, t)
        loss = F.mse_loss(v.float(), eps - r0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, p in zip(ema.parameters(), G.parameters()):
                pe.lerp_(p, 1.0 - a.ema)
        acc += loss.item()
        n_acc += 1

        if (step + 1) % a.log_every == 0:
            print(f"[{step + 1}/{a.steps}] loss {acc / n_acc:.4f}  lr {lr_at(step):.2e}  "
                  f"{(step + 1 - start) / (time.time() - t0):.1f} it/s", flush=True)
            acc, n_acc = 0.0, 0
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            v = validate(ema, val, cfg, margin, dev, a.val_n, amp, cfg["steps"])
            tag = ""
            if v["loss"] < best:
                best = v["loss"]
                save(os.path.join(out, "best.pt"), ema)
                tag = "  *best*"
            print(f"[val {step + 1}] loss {v['loss']:.4f}   {cfg['steps']}-step samples / teacher: "
                  f"fine texture {v['hf']:.2f}, slope {v['slope']:.2f}   (rf {cfg.get('rf', '?')} px){tag}", flush=True)
            logf.write(json.dumps(dict(step=step + 1, **v)) + "\n")
            logf.flush()
        stop = bool(a.max_minutes) and (time.time() - t0) / 60.0 >= a.max_minutes
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or stop:
            save(os.path.join(out, "last.pt"), ema)
            torch.save(dict(G=G.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), step=step + 1, best=best,
                            cfg=cfg), state_path)
        if stop and step + 1 < a.steps:
            print(f"[time limit] stopped at step {step + 1}/{a.steps}; state saved in {out} -- run the same command "
                  f"again to resume", flush=True)
            return
    print(f"[done] best val loss {best:.4f}. Checkpoints in {out} (best.pt, last.pt); halo for {cfg['steps']} steps: "
          f"{stm.halo_coarse(cfg, cfg['steps'])} coarse px.")


if __name__ == "__main__":
    main()
