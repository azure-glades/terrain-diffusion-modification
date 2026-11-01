#!/usr/bin/env python3
"""Level-by-level cascade picture that REUSES ONE student checkpoint at every level.

Same output layout as show_levels.py (rows = levels, columns = teacher | cascade | on-teacher-input),
but instead of one checkpoint per level it loads a single checkpoint and applies it at levels
start_level+1 .. levels-1. This is a DOMAIN-SHIFT PROBE, not a quality result: the weights were
trained for their own level's input statistics / residual scale / noise grid.

  python show_levels_shared.py --data data/val2 --student runs/l4/best.pt \
      --sample 4,5 --out img/levels_shared.png

  # optionally retarget the residual scale to each target level so the added detail is not wildly mis-scaled:
  python show_levels_shared.py --data data/val2 --student runs/l4/best.pt \
      --res-std-from data/train2 --sample 4,5 --out img/levels_shared_rs.png
"""
import argparse
import os
import sys

import numpy as np

from eval_terrain import student_candidate
from make_pyramid_data import PyramidData, hillshade, load_manifest


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--student", required=True, help="ONE checkpoint reused at every level")
    ap.add_argument("--sample", default="0", help="patch index, or comma-separated list")
    ap.add_argument("--out", default="levels_shared.png")
    ap.add_argument("--start-level", type=int, default=0, help="cascade starts from the teacher's level k")
    ap.add_argument("--res-std-from", default=None,
                    help="dataset dir to recalibrate cfg['res_std'] per target level (default: keep the "
                         "checkpoint's own res_std, which is usually wrong for other levels)")
    ap.add_argument("--win", type=int, default=24, help="window width in level-0 pixels (tile = win * 2^(levels-1) px)")
    ap.add_argument("--x0", type=int, default=None, help="window left edge in level-0 px (default: centred)")
    ap.add_argument("--y0", type=int, default=None, help="window top edge in level-0 px (default: centred)")
    ap.add_argument("--vexag", type=float, default=2.0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def shade(z, box, mpp, vexag, pad=2):
    """Hillshade of z restricted to box=(y0,x0,h,w), computed with a small apron so tile edges are clean."""
    y0, x0, h, w = box
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    yb, xb = min(y0 + h + pad, z.shape[0]), min(x0 + w + pad, z.shape[1])
    hs = hillshade(z[ya:yb, xa:xb].astype(np.float64), mpp, vexag)
    return (hs[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] * 255).astype(np.uint8)


def make_shared_gens(student_path, need, a, res_std_by_level=None):
    """Load ONE checkpoint and return {L: gen} where gen is the same weights with cfg['level']=L."""
    import student_model as sm
    import torch

    base_gen = student_candidate(student_path, a.device, a.noise_seed, a.dtype)
    model, cfg0, dev = base_gen.model, dict(base_gen.cfg), base_gen.device
    amp = torch.bfloat16 if a.dtype == "bf16" else None

    gens = {}
    for L in need:
        cfg = dict(cfg0)
        cfg["level"] = L                                   # noise grid + any level-dependent cfg follows target
        if res_std_by_level is not None and L in res_std_by_level:
            cfg["res_std"] = res_std_by_level[L]           # retarget the residual amplitude

        def gen(coarse, origin=(0, 0), _cfg=cfg):
            return sm.student_generate(model, _cfg, coarse, origin, seed=a.noise_seed, device=dev, dtype=amp)

        gen.model, gen.cfg, gen.device = model, cfg, dev
        gens[L] = gen
    return gens


def calibrate_res_std(root, levels, need, n=64):
    """Mean residual std per level, sampled from a dataset (encoded space, matching training)."""
    ds = PyramidData(root, levels=levels, augment=False, space="sqrt", norm=50.0, preload=False)
    rng = np.random.default_rng(0)
    out = {}
    for L in need:
        idx = rng.integers(0, len(ds), min(n, len(ds)))
        out[L] = float(np.mean([ds[int(i)]["residuals"][L].std() for i in idx]))
    return out


def main():
    a = parse()
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("needs Pillow: pip install pillow")
    man = load_manifest(a.data)
    levels = man["levels"]
    need = list(range(a.start_level + 1, levels))
    if not need:
        sys.exit(f"--start-level {a.start_level} leaves no levels to generate (levels={levels})")

    res_std_by_level = None
    if a.res_std_from:
        res_std_by_level = calibrate_res_std(a.res_std_from, levels, need)
        print("[res_std] per-level (encoded space): " + ", ".join(f"L{L}={v:.4f}" for L, v in res_std_by_level.items()))

    gens = make_shared_gens(a.student, need, a, res_std_by_level)
    print(f"[shared] using {a.student} at levels {need} (weights identical, cfg retargeted per level)")

    ds = PyramidData(a.data, levels=levels, augment=False, space="metres", norm=1.0)
    tile = a.win * 2 ** (levels - 1)
    base, ext = os.path.splitext(a.out)
    samples = [int(s) for s in a.sample.split(",")]
    for si in samples:
        if not 0 <= si < len(ds):
            sys.exit(f"--sample {si} out of range (dataset has {len(ds)} patches)")
        T = [x[0].astype(np.float32) for x in ds[si]["levels"]]
        h0, w0 = T[0].shape
        y0 = (h0 - a.win) // 2 if a.y0 is None else a.y0
        x0 = (w0 - a.win) // 2 if a.x0 is None else a.x0
        if not (0 <= y0 <= h0 - a.win and 0 <= x0 <= w0 - a.win):
            sys.exit(f"window ({y0},{x0}) size {a.win} falls outside the {h0}x{w0} level-0 patch")
        casc, orc = {a.start_level: T[a.start_level]}, {}
        for L in need:
            casc[L] = gens[L](casc[L - 1], (0, 0))
            orc[L] = gens[L](T[L - 1], (0, 0))

        lab_w, head_h = 120, 30
        sheet = np.full((head_h + levels * tile, lab_w + 3 * tile), 40, np.uint8)
        grey = np.full((tile, tile), 90, np.uint8)
        for L in range(levels):
            f = 2 ** L                                        # level-L px per level-0 px
            mpp = man["mpp"] * 2 ** (levels - 1 - L)
            box = (y0 * f, x0 * f, a.win * f, a.win * f)
            k = tile // (a.win * f)                           # integer upscale so every tile has the same size
            cols = []
            for src in (T[L], casc.get(L), orc.get(L)):
                if src is None or (src is casc.get(L) and L < a.start_level):
                    cols.append(grey)
                else:
                    cols.append(np.repeat(np.repeat(shade(src, box, mpp, a.vexag), k, 0), k, 1))
            if L <= a.start_level:                            # nothing generated here: the teacher is the input
                cols[2] = grey
                if L < a.start_level:
                    cols[1] = grey
            r = head_h + L * tile
            for c, im in enumerate(cols):
                sheet[r:r + tile, lab_w + c * tile:lab_w + (c + 1) * tile] = im
        img = Image.fromarray(sheet).convert("RGB")
        d = ImageDraw.Draw(img)
        for c, name in enumerate(("teacher", "cascade (student input = student)", "student on teacher input")):
            d.text((lab_w + c * tile + 6, 9), name, fill=(255, 255, 255))
        for L in range(levels):
            mpp = man["mpp"] * 2 ** (levels - 1 - L)
            d.text((8, head_h + L * tile + 8), f"level {L}", fill=(255, 255, 255))
            d.text((8, head_h + L * tile + 24), f"{mpp:.0f} m/px", fill=(200, 200, 200))
            d.text((8, head_h + L * tile + 40), f"{a.win * 2 ** L} px", fill=(200, 200, 200))
        path = a.out if len(samples) == 1 else f"{base}_{si}{ext or '.png'}"
        img.save(path)
        print(f"[png] wrote {path}  patch {si}, window level-0 px ({y0},{x0}) size {a.win}; "
              f"grey = not generated (teacher is the input at that level)")


if __name__ == "__main__":
    main()