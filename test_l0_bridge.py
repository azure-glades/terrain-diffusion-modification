#!/usr/bin/env python3
"""Can a level-1 student bridge coarser-than-480 m maps up to a 480 m level 0, zero-shot?

Ground truth: the teacher's level 0 (480 m/px) of a validation patch.
For k = 1..4: block-average it k more times (k=4 -> 7.7 km/px, the teacher's coarse-stage scale), then apply the SAME
network k times (each step 2x finer, with its own residual scale measured from the data) back to 480 m, and compare
with the truth and with plain bilinear upsampling.

  python test_l0_bridge.py --data data/val3 --student runs/l1_gan/last.pt --sample 0,3,7 --out bridge.png

Notes: a patch is only 128 px at level 0, so k=4 starts from an 8x8 map (zero padding dominates); read it as a
rough indication. Residual scales come from the data itself (one scalar per step, no training).
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import student_model as sm  # noqa: E402
from make_pyramid_data import (PyramidData, build_pyramid, decode_elev, down2, encode_elev, hillshade,  # noqa: E402
                               load_manifest, up2)


def box(x, k):
    p = np.pad(x, k // 2, mode="edge")
    c = np.cumsum(np.cumsum(np.pad(p, ((1, 0), (1, 0))), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def stats(z, ref):
    hp = lambda a: a - box(a, 5)                                   # noqa: E731
    sl = lambda a: np.hypot(*np.gradient(a))                       # noqa: E731
    return dict(mae=float(np.abs(z - ref).mean()),
                hf=float(hp(z).std() / max(hp(ref).std(), 1e-9)),
                slope=float(sl(z).mean() / max(sl(ref).mean(), 1e-9)),
                std=float(z.std() / max(ref.std(), 1e-9)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--student", required=True, help="dedicated level-1 checkpoint (e.g. runs/l1_gan/last.pt)")
    ap.add_argument("--sample", default="0", help="patch index or comma list")
    ap.add_argument("--ks", default="1,2,3,4", help="how many halvings below level 0 to start from")
    ap.add_argument("--calib", type=int, default=32, help="patches used to measure the residual scales")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--vexag", type=float, default=2.0)
    ap.add_argument("--out", default="bridge.png")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    import torch
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg = sm.load_student(a.student, dev)
    if cfg.get("shared"):
        sys.exit("--student must be a dedicated checkpoint")
    ds = PyramidData(a.data, levels=5, crop=None, augment=False, preload=True)
    man = load_manifest(a.data)
    ks = [int(k) for k in a.ks.split(",")]
    kmax = max(ks)
    space, norm = cfg["space"], cfg["norm"]
    print(f"[bridge] net: level-1 student {a.student}, encoding {space}/{norm}, trained res_std {cfg['res_std']:.4f}")

    # residual scale for each step j (coarse = level0 averaged j times -> finer = averaged j-1 times)
    n = min(a.calib, len(ds))
    acc = {j: [] for j in range(1, kmax + 1)}
    for i in range(n):
        l0 = build_pyramid(ds._pre["elev"][i].astype(np.float32), 5)[0]
        lv = [l0]
        for _ in range(kmax):
            lv.append(down2(lv[-1]))
        enc = [encode_elev(z, space, norm) for z in lv]
        for j in range(1, kmax + 1):
            acc[j].append(float((enc[j - 1] - up2(enc[j])).std()))
    res = {j: float(np.mean(v)) for j, v in acc.items()}
    print("[bridge] residual std per step (step j: level0/2^j -> level0/2^(j-1)): "
          + ", ".join(f"j={j}: {v:.4f}" for j, v in res.items()) + f"   (level-1 training value {cfg['res_std']:.4f})")

    sample = [int(s) for s in a.sample.split(",")]
    rows_img, table = [], {k: [] for k in ks}
    mpp0 = man["mpp"] * 2 ** 4
    for si in sample:
        l0 = build_pyramid(ds._pre["elev"][si].astype(np.float32), 5)[0]
        lv = [l0]
        for _ in range(kmax):
            lv.append(down2(lv[-1]))
        tiles = [("teacher L0", l0)]
        for k in ks:
            z = lv[k]
            bil = z
            for _ in range(k):
                bil = up2(bil)
            for j in range(k, 0, -1):                                    # step j: level0/2^j -> level0/2^(j-1)
                c = dict(cfg, res_std=res[j], level=100 + j, noise_on=True, shared=False)
                with torch.no_grad():
                    z = sm.student_generate(model, c, z, (0, 0), seed=a.noise_seed, device=dev)
            table[k].append((stats(z, l0), stats(bil, l0)))
            tiles += [(f"k={k} bilinear", bil), (f"k={k} bridge", z)]
        rows_img.append(tiles)

    print(f"\n{'start':>14} | {'bridge MAE':>10} {'hf':>6} {'slope':>6} {'std':>6} | {'bilinear MAE':>12} {'hf':>6} {'slope':>6} {'std':>6}"
          "   (ratios vs teacher, 1.0 = same character)")
    for k in ks:
        b = {m: np.mean([t[0][m] for t in table[k]]) for m in ("mae", "hf", "slope", "std")}
        l = {m: np.mean([t[1][m] for t in table[k]]) for m in ("mae", "hf", "slope", "std")}
        print(f"{f'k={k} ({480 * 2 ** k / 1000:.1f} km)':>14} | {b['mae']:10.1f} {b['hf']:6.2f} {b['slope']:6.2f} {b['std']:6.2f} |"
              f" {l['mae']:12.1f} {l['hf']:6.2f} {l['slope']:6.2f} {l['std']:6.2f}")

    try:
        from PIL import Image, ImageDraw
        S = 192
        out_rows = []
        for tiles in rows_img:
            ims = []
            for name, z in tiles:
                hs = hillshade(z.astype(np.float64), mpp0, a.vexag)
                im = Image.fromarray((hs * 255).astype(np.uint8)).resize((S, S), Image.NEAREST).convert("RGB")
                ImageDraw.Draw(im).text((4, 4), name, fill=(255, 255, 0))
                ims.append(np.asarray(im))
            out_rows.append(np.concatenate(ims, 1))
        Image.fromarray(np.concatenate(out_rows, 0)).save(a.out)
        print(f"wrote {a.out} (columns: teacher L0, then bilinear | bridge for each k)")
    except ImportError:
        print("PIL missing: no image written")


if __name__ == "__main__":
    main()