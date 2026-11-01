"""Chain the per-level students (level 0 -> finest) and compare against the teacher.

For every level L it reports three things against the teacher's level L:
  bilinear : plain 2x upsampling all the way from level 0 (the floor)
  oracle   : student L fed the TEACHER's level L-1   (what training measured)
  cascade  : student L fed the previous student's output (what you will actually run)
cascade vs oracle shows how much error compounds across levels ("exposure bias").
Level 0 is the teacher's (downsampled) level 0, so this tests levels 1..N only; a procedural level 0 comes later.

  python cascade_eval.py --data data/val2 --students runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt \
      --samples 8 --png cascade.png
"""
import argparse
import sys
import time

import numpy as np

from eval_terrain import aggregate, fmt, montage, sample_stats, student_candidate
from make_pyramid_data import PyramidData, load_manifest, up2


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="validation dir with LARGE patches (e.g. data/val2, 2048 px)")
    ap.add_argument("--students", required=True, help="comma-separated checkpoints for levels 1..N (any order)")
    ap.add_argument("--levels", type=int, default=None)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--start-level", type=int, default=0,
                    help="begin the chain from the TEACHER's level k instead of 0 (shows how much the result depends on "
                         "how much structure the coarse input already has)")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--png", default=None, help="montage of the finest level: bilinear | cascade | teacher")
    return ap.parse_args()


def main():
    a = parse()
    man = load_manifest(a.data)
    levels = a.levels or man["levels"]
    gens = {}
    for p in a.students.split(","):
        g = student_candidate(p.strip(), a.device, a.noise_seed, a.dtype)
        gens[g.cfg["level"]] = g
    need = list(range(a.start_level + 1, levels))
    if any(L not in gens for L in need):
        sys.exit(f"need a student for every level {need}; got levels {sorted(gens)}")
    rf = {L: int(gens[L].cfg.get("rf", 32)) for L in need}
    margin = {a.start_level: 0}                      # border (in that level's pixels) contaminated by zero padding
    for L in need:
        margin[L] = 2 * margin[L - 1] + rf[L]
    print(f"[cascade] chain starts from the teacher's level {a.start_level}")
    print("[cascade] receptive-field radius per level (px): " + ", ".join(f"L{L}={rf[L]}" for L in need))
    print("[cascade] border excluded from the scores (px): " + ", ".join(f"L{L}={margin[L]}" for L in need))

    ds = PyramidData(a.data, levels=levels, augment=False, space="metres", norm=1.0)
    idx = np.linspace(0, len(ds) - 1, min(a.samples, len(ds))).astype(int)
    rows = {(L, k): [] for L in need for k in ("bilinear", "oracle", "cascade")}
    keep, t_gen, mpx = [], 0.0, 0.0
    for n, i in enumerate(idx):
        T = [x[0].astype(np.float32) for x in ds[int(i)]["levels"]]       # teacher, coarse -> fine
        if min(T[a.start_level].shape) <= 2 * margin[levels - 1]:
            print(f"[warn] patch {T[0].shape} at level 0 is small; use larger patches (--patch 2048)")
        casc, bil = T[a.start_level], T[a.start_level]
        outs = {}
        for L in need:
            t0 = time.time()
            casc = gens[L](casc, (0, 0))                                   # chained: its own output feeds the next level
            t_gen += time.time() - t0
            orc = gens[L](T[L - 1], (0, 0))
            bil = up2(bil)
            m = margin[L]
            sl = (slice(m, T[L].shape[0] - m), slice(m, T[L].shape[1] - m))
            mpp = man["mpp"] * 2 ** (levels - 1 - L)
            for k, p in (("bilinear", bil), ("oracle", orc), ("cascade", casc)):
                rows[(L, k)].append(sample_stats(T[L][sl].astype(np.float64), p[sl].astype(np.float64), mpp))
            outs[L] = (bil, casc)
        mpx += casc.size / 1e6
        if n < 4:
            keep.append((T[levels - 2], T[levels - 1], outs[levels - 1]))
        print(f"[{n + 1}/{len(idx)}] patch done", flush=True)

    print("\n== per level: RMSE (m) vs teacher, and terrain statistics of the cascade output / teacher ==")
    table = []
    for L in need:
        b, o, c = (aggregate(rows[(L, k)]) for k in ("bilinear", "oracle", "cascade"))
        table.append([f"L{L} ({man['mpp'] * 2 ** (levels - 1 - L):.0f} m)", f"{b['rmse']:.2f}", f"{o['rmse']:.2f}",
                      f"{c['rmse']:.2f}", f"{100 * (1 - c['rmse'] / max(b['rmse'], 1e-9)):.1f}%",
                      f"{c['s90_p'] / max(c['s90_t'], 1e-12):.2f}", f"{c['s99_p'] / max(c['s99_t'], 1e-12):.2f}",
                      f"{c['rough_p'] / max(c['rough_t'], 1e-12):.2f}", f"{c['hf_p'] / max(c['hf_t'], 1e-12):.2f}"])
    fmt(table, ["level", "bilinear", "oracle", "cascade", "casc gain", "slope p90", "slope p99", "roughness", "HF energy"])
    print("\n== same statistics for the ORACLE (student fed the teacher's previous level), oracle/teacher ==")
    fmt([[f"L{L}"] + [f"{v:.2f}" for v in (lambda o: (o['s90_p'] / max(o['s90_t'], 1e-12), o['s99_p'] / max(o['s99_t'], 1e-12),
                                                    o['rough_p'] / max(o['rough_t'], 1e-12), o['hf_p'] / max(o['hf_t'], 1e-12)))(
                                      aggregate(rows[(L, 'oracle')]))] for L in need],
        ["level", "slope p90", "slope p99", "roughness", "HF energy"])
    print("(cascade >> oracle RMSE = errors compound; statistics columns are cascade/teacher, 1.00 = match.)")
    print(f"\n[speed] student cascade (levels 1..{levels - 1}): {1000 * t_gen / max(mpx, 1e-9):.0f} ms per Mpx of finest output "
          f"(includes CPU noise hashing and transfers; level 0 not included)")
    if a.png:
        pairs = [(c, f) for c, f, _ in keep]
        outs = {"bilinear": [o[0] for _, _, o in keep], "cascade": [o[1] for _, _, o in keep]}
        montage(a.png, pairs, outs, man["mpp"], ["bilinear", "cascade"])


if __name__ == "__main__":
    main()