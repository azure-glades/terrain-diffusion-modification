#!/usr/bin/env python3
"""Look at the structure model: teacher level 0 (480 m) next to maps sampled from the teacher's coarse map.

  python show_structure.py --data data/val3 --ckpt runs/struct/best.pt --out structure.png      # montage + statistics
  python show_structure.py --ckpt runs/struct/best.pt --seam                                    # statelessness check

Montage columns: coarse elevation upsampled (what the model is given) | teacher | sample (seed 0) | sample (seed 1).
The statistics compare samples with the teacher on the patch centre (1.00 = match); they are distribution
statistics, not errors: a sample is a different terrain with the same coarse layout, so RMSE is not meaningful.

--seam generates one large window and two overlapping halves of it and reports how far from a window edge the
outputs still differ: that distance is the halo the sampler really needs (compare with the default the explorer
uses, steps x receptive field). No dataset needed.
"""
import argparse
import sys

import numpy as np
import torch

import structure_model as stm
from eval_terrain import aggregate, sample_stats
from make_pyramid_data import StructData, decode_elev, hillshade


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default=None, help="dataset with coarse sidecars (2048 px patches)")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--steps", type=int, default=None, help="sampler steps (default: the checkpoint's)")
    ap.add_argument("--out", default="structure.png")
    ap.add_argument("--seam", action="store_true")
    ap.add_argument("--seam-size", type=int, default=40, help="coarse px of the big window in the seam test")
    ap.add_argument("--tol", type=float, default=0.5, help="seam tolerance in metres")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    return ap.parse_args()


def seam(model, cfg, a, dev, amp):
    """Window-independence of the K-step sampler, measured (not assumed)."""
    n, f, steps = a.seam_size, cfg["cpp"], a.steps or cfg["steps"]
    rng = np.random.default_rng(0)
    c = rng.normal(size=(stm.N_COND, n, n)).astype(np.float32)
    for _ in range(3):                                            # smooth random coarse map in plausible units
        c = (c + np.roll(c, 1, 1) + np.roll(c, -1, 1) + np.roll(c, 1, 2) + np.roll(c, -1, 2)) / 5
    c = c / c.std(axis=(1, 2), keepdims=True) * np.asarray(stm.COARSE_STD, np.float32)[:, None, None] \
        + np.asarray(stm.COARSE_MEAN, np.float32)[:, None, None]
    gen = lambda cc, o: stm.structure_generate(model, cfg, cc, o, 0, steps, dev, amp)   # noqa: E731
    wa = int(n * 0.7)
    full = gen(c, (0, 0))
    left = gen(np.ascontiguousarray(c[:, :, :wa]), (0, 0))
    right = gen(np.ascontiguousarray(c[:, :, n - wa:]), (0, n - wa))
    ea = np.abs(left - full[:, :f * wa]).max(axis=0)              # error per column, vs distance to the cut edge
    eb = np.abs(right - full[:, f * (n - wa):]).max(axis=0)
    d = np.concatenate([np.arange(f * wa)[::-1], np.arange(f * wa)])
    e = np.concatenate([ea, eb])
    need = next((d0 for d0 in range(int(d.max()) + 1) if e[d >= d0].max() <= a.tol), None)
    dflt = stm.halo_coarse(cfg, steps)
    print(f"[seam] {steps} steps, per-step receptive field {cfg.get('rf', '?')} px: outputs agree to {a.tol} m beyond "
          + (f"{need} px = {-(-need // f)} coarse px from a window edge" if need is not None else "NO distance tested")
          + f"; explorer default halo = {dflt} coarse px ({dflt * f} px)")
    for lo, hi in ((0, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 10 ** 9)):
        sel = (d >= lo) & (d < hi)
        if sel.any():
            print(f"   {lo:4d}-{min(hi, int(d.max()) + 1):4d} px from the edge: max difference {e[sel].max():9.3f} m")
    return need


def main():
    a = parse()
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = torch.bfloat16 if a.dtype == "bf16" else None
    model, cfg = stm.load_structure(a.ckpt, dev)
    steps = a.steps or cfg["steps"]
    if a.seam:
        seam(model, cfg, a, dev, amp)
    if not a.data:
        if not a.seam:
            sys.exit("pass --data for the montage and/or --seam")
        return
    ds = StructData(a.data, space=cfg["space"], norm=cfg["norm"], augment=False)
    if not len(ds):
        sys.exit(f"{a.data} has no usable patches (needs 2048 px patches with coarse sidecars)")
    f, M, S, mpp = cfg["cpp"], ds.margin, ds.S, cfg["mpp"]
    o = M * f
    rows, tiles = [], []
    for i in np.linspace(0, len(ds) - 1, min(a.samples, len(ds))).astype(int):
        it = ds[int(i)]
        teacher = decode_elev(it["x"][0], cfg["space"], cfg["norm"])
        cut = lambda z: z[o:o + S, o:o + S]                         # noqa: E731
        base = cut(stm.decode_elev_t(stm.make_cond(torch.from_numpy(it["coarse"])[None], cfg)[1], cfg["norm"])[0, 0].numpy())
        outs = [cut(stm.structure_generate(model, cfg, it["coarse"], (0, 0), seed, steps, dev, amp)) for seed in (0, 1)]
        q = S // 4                                                 # centre: away from the (too small) stored margin
        for z in outs:
            rows.append(sample_stats(teacher[q:-q, q:-q].astype(np.float64), z[q:-q, q:-q].astype(np.float64), mpp))
        tiles.append(np.concatenate([(hillshade(z, mpp) * 255).astype(np.uint8) for z in (base, teacher, *outs)], axis=1))
    m = aggregate(rows)
    r = lambda k: m[f"{k}_p"] / max(m[f"{k}_t"], 1e-12)             # noqa: E731
    print(f"[stats] {len(rows)} samples, {steps} steps; sample / teacher:  slope p50 {r('s50'):.2f}  p90 {r('s90'):.2f}  "
          f"p99 {r('s99'):.2f}  roughness {r('rough'):.2f}  fine-detail energy {r('hf'):.2f}")
    try:
        from PIL import Image
        Image.fromarray(np.concatenate(tiles, axis=0)).save(a.out)
        print(f"[png] wrote {a.out}  (columns: coarse input | teacher | sample seed 0 | sample seed 1; {S} px = {S * mpp / 1000:.0f} km)")
    except ImportError:
        print("[note] Pillow not installed; skipping the montage")


if __name__ == "__main__":
    main()
