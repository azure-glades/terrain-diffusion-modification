"""Level-by-level picture of the cascade.

One row per level, three columns (hillshaded, same ground area in every tile):
  teacher  : the real level L from the teacher
  cascade  : student L fed the previous STUDENT's output  (what you would really run)
  on teacher input : student L fed the TEACHER's level L-1  (best case for that student)

  python show_levels.py --data data/val2 --students runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt \
      --sample 0,3 --out levels.png          # writes levels_0.png, levels_3.png
"""
import argparse
import os
import sys

import numpy as np

from eval_terrain import student_candidates
from make_pyramid_data import PyramidData, hillshade, load_manifest


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--students", required=True, help="comma-separated checkpoints, one per level 1..N")
    ap.add_argument("--sample", default="0", help="patch index, or comma-separated list")
    ap.add_argument("--out", default="levels.png")
    ap.add_argument("--start-level", type=int, default=0, help="cascade starts from the teacher's level k")
    ap.add_argument("--win", type=int, default=24, help="window width in level-0 pixels (tile = win * 2^(levels-1) px)")
    ap.add_argument("--x0", type=int, default=None, help="window left edge in level-0 px (default: centred)")
    ap.add_argument("--y0", type=int, default=None, help="window top edge in level-0 px (default: centred)")
    ap.add_argument("--vexag", type=float, default=2.0)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--reuse-net", default=None,
                    help="ZERO-SHOT: run this dedicated checkpoint's network at EVERY level, taking each level's residual "
                         "scale from the matching --students checkpoint (e.g. runs/l1_gan/last.pt)")
    ap.add_argument("--device", default=None)
    return ap.parse_args()


def shade(z, box, mpp, vexag, pad=2):
    """Hillshade of z restricted to box=(y0,x0,h,w), computed with a small apron so tile edges are clean."""
    y0, x0, h, w = box
    ya, xa = max(y0 - pad, 0), max(x0 - pad, 0)
    yb, xb = min(y0 + h + pad, z.shape[0]), min(x0 + w + pad, z.shape[1])
    hs = hillshade(z[ya:yb, xa:xb].astype(np.float64), mpp, vexag)
    return (hs[y0 - ya:y0 - ya + h, x0 - xa:x0 - xa + w] * 255).astype(np.uint8)


def main():
    a = parse()
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        sys.exit("needs Pillow: pip install pillow")
    man = load_manifest(a.data)
    levels = man["levels"]
    gens = {}
    for p in a.students.split(","):
        for g in student_candidates(p.strip(), a.device, a.noise_seed, a.dtype, reuse=a.reuse_net):
            gens[g.cfg["level"]] = g
    need = list(range(a.start_level + 1, levels))
    if any(L not in gens for L in need):
        sys.exit(f"need a student for every level {need}; got levels {sorted(gens)}")
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