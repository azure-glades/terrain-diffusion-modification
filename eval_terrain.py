#!/usr/bin/env python3
"""Evaluation harness for coarse-to-fine terrain generators (students) against teacher data.

A *candidate* is a function  gen(coarse_metres (h,w), origin=(ci,cj)) -> fine_metres (2h,2w):
the level-L generator given the level-(L-1) map. Built-ins:  bilinear  |  student:<checkpoint.pt>

  python eval_terrain.py compare --data data/val --level 4 --candidates bilinear,student:runs/l4/best.pt \
                                 --samples 32 --png compare.png --seam --bench
  python eval_terrain.py selftest          # numpy-only checks with synthetic data (no GPU/model)

What it reports
  1. Paired fidelity vs the teacher (MAE/RMSE in metres, gradient error, "detail gain" = RMSE reduction vs bilinear).
     Meaningful for deterministic students; a stochastic student can legitimately differ from the teacher sample.
  2. Terrain statistics, candidate vs teacher (ratio 1.0 is ideal): slope percentiles, Laplacian roughness,
     radial power-spectrum slope and high-frequency energy. These are what show blur / over-sharpening.
  3. --seam : STATELESSNESS test. Generate the same region as one big window and as two overlapping windows
     (with absolute-coordinate noise); error vs distance from the interior window edge gives the halo you need.
  4. --bench: speed of the student (model-only and end-to-end) in ms per output megapixel.
  5. --png  : hillshade montage  [bilinear | candidates... | teacher].
Reference points worth keeping: `bilinear` (zero detail = floor), the one-step teacher (quality you rejected), the teacher.
"""
import argparse
import collections
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pyramid_data import PyramidData, hillshade, load_manifest, up2  # noqa: E402

# =============================================================== statistics


def radial_psd(x, nbins=24):
    x = x - x.mean()
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    f = np.fft.fftshift(np.fft.fft2(x * win))
    p = (np.abs(f) ** 2) / max((win ** 2).sum(), 1e-12)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = np.linspace(0, 0.5, nbins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    ok = (idx >= 0) & (idx < nbins)
    s = np.bincount(idx[ok], weights=p.ravel()[ok], minlength=nbins)
    n = np.bincount(idx[ok], minlength=nbins)
    return 0.5 * (edges[1:] + edges[:-1]), s / np.maximum(n, 1)


def spectrum_stats(x):
    k, p = radial_psd(x)
    m = (k > 0.03) & (p > 0)
    slope = float(np.polyfit(np.log10(k[m]), np.log10(p[m]), 1)[0])
    hf = float(p[k >= 0.25].sum())
    return slope, hf


def lap(z):
    return 4 * z[1:-1, 1:-1] - z[:-2, 1:-1] - z[2:, 1:-1] - z[1:-1, :-2] - z[1:-1, 2:]


def sample_stats(t, p, mpp):
    d = {}
    e = p - t
    d["mae"], d["mse"] = float(np.abs(e).mean()), float((e ** 2).mean())
    gyt, gxt = np.gradient(t)
    gyp, gxp = np.gradient(p)
    d["gmae"] = 0.5 * float(np.abs(gxp - gxt).mean() + np.abs(gyp - gyt).mean())
    mt, mp = np.hypot(gxt, gyt) / mpp, np.hypot(gxp, gyp) / mpp
    for q in (50, 90, 99):
        d[f"s{q}_t"], d[f"s{q}_p"] = float(np.percentile(mt, q)), float(np.percentile(mp, q))
    d["rough_t"], d["rough_p"] = float(lap(t).std()), float(lap(p).std())
    (d["slope_t"], d["hf_t"]), (d["slope_p"], d["hf_p"]) = spectrum_stats(t), spectrum_stats(p)
    return d


def aggregate(rows):
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    m["rmse"] = float(np.sqrt(m["mse"]))
    return m


# =============================================================== seam / statelessness test


def seam_test(gen, coarse, tol=0.01, frac=0.69, floor_max=0.05):
    """gen(coarse_window, (ci, cj)) -> fine window. Compares two overlapping column windows with one big window."""
    h, w = coarse.shape
    wa = (int(w * frac) // 2) * 2
    full = gen(coarse, (0, 0))
    a = gen(np.ascontiguousarray(coarse[:, :wa]), (0, 0))
    b = gen(np.ascontiguousarray(coarse[:, w - wa:]), (0, w - wa))
    err_a = np.abs(a - full[:, :2 * wa]).max(axis=0)                # per fine column
    err_b = np.abs(b - full[:, 2 * (w - wa):]).max(axis=0)
    d = np.concatenate([np.arange(2 * wa)[::-1], np.arange(2 * wa)])   # distance to the INTERIOR edge of each window
    e = np.concatenate([err_a, err_b])
    bands = [(0, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10 ** 9)]
    table = []
    for lo, hi in bands:
        sel = (d >= lo) & (d < hi)
        if sel.any():
            table.append((lo, min(hi, int(d.max()) + 1), float(e[sel].max()), float(e[sel].mean())))
    # Far from any window edge the error should be ~0. A tiny constant floor (cm level) is numerical: different
    # window shapes make cuDNN pick different kernels / TF32 rounding. It is accepted only if it is below floor_max;
    # a large constant error (window-dependent output) is NOT accepted and is reported as not stateless.
    floor = float(e[d >= 0.5 * d.max()].max())
    eff_tol = max(tol, 2.0 * floor) if floor <= floor_max else tol
    halo = None                                  # None = error never falls below tolerance (not stateless)
    for d0 in range(int(d.max()) + 1):
        if e[d >= d0].max() <= eff_tol:
            halo = d0
            break
    return dict(halo_fine_px=halo, table=table, worst=float(e.max()), tol=tol, floor=floor, eff_tol=eff_tol)


# =============================================================== candidates


def bilinear_candidate():
    return lambda coarse, origin=(0, 0): up2(np.asarray(coarse, np.float32))


def student_candidates(path, device=None, seed=0, dtype=None, tf32=False, reuse=None):
    """Loads one checkpoint and returns one generator per level it covers: a single one for a dedicated student,
    several (sharing one network) for a shared student.

    reuse=<other checkpoint>: ZERO-SHOT test. The network comes from `reuse` (e.g. the level-1 GAN student) while the
    level number, residual scale (res_std) and space come from `path`; the reused network always gets hashed noise."""
    import student_model as sm
    import torch
    torch.backends.cudnn.allow_tf32 = bool(tf32)          # cuDNN allows TF32 by default on Ampere; off = exact fp32
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg0 = sm.load_student(path, dev)
    amp = torch.bfloat16 if dtype == "bf16" else None
    levels_cfg = sm.level_cfgs(cfg0)
    if reuse:
        model, ncfg = sm.load_student(reuse, dev)
        if ncfg.get("shared"):
            raise ValueError("--reuse-net must be a dedicated (not shared) student")
        for c in levels_cfg:
            if (c["space"], c["norm"]) != (ncfg["space"], ncfg["norm"]):
                raise ValueError(f"{path} and {reuse} use different encodings")
        levels_cfg = [dict(c, n_noise=ncfg["n_noise"], noise_on=True, shared=False, rf=int(ncfg.get("rf", 15)))
                      for c in levels_cfg]
    gens = []
    for cfg in levels_cfg:
        def gen(coarse, origin=(0, 0), cfg=cfg):
            return sm.student_generate(model, cfg, coarse, origin, seed=seed, device=dev, dtype=amp)
        gen.model, gen.cfg, gen.device = model, cfg, dev
        gens.append(gen)
    return gens


def student_candidate(path, device=None, seed=0, dtype=None, tf32=False):
    return student_candidates(path, device, seed, dtype, tf32)[0]


def make_candidates(spec, a):
    out = collections.OrderedDict()
    for s in spec.split(","):
        s = s.strip()
        if s == "bilinear":
            out[s] = bilinear_candidate()
        elif s.startswith("student:"):
            out[s] = student_candidate(s.split(":", 1)[1], a.device, a.noise_seed, a.dtype, a.tf32)
        elif s:
            sys.exit(f"unknown candidate {s!r} (use bilinear or student:<path>)")
    if "bilinear" not in out:
        out["bilinear"] = bilinear_candidate()
        out.move_to_end("bilinear", last=False)
    return out


# =============================================================== comparison


def compare(pairs, cands, mpp):
    """pairs: list of (coarse, fine). Returns {name: aggregate dict}, plus target-reference stats."""
    rows = {n: [] for n in cands}
    outs = {}
    for si, (coarse, fine) in enumerate(pairs):
        for n, g in cands.items():
            p = g(coarse, (0, 0)).astype(np.float64)
            rows[n].append(sample_stats(fine.astype(np.float64), p, mpp))
            if si < 4:
                outs.setdefault(n, []).append(p)
    return {n: aggregate(r) for n, r in rows.items()}, outs


def fmt(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    print(line(header))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))


def print_comparison(res):
    base_rmse = res["bilinear"]["rmse"]
    print("\n== paired fidelity vs teacher (lower error is better) ==")
    fmt([[n, f"{r['mae']:.2f}", f"{r['rmse']:.2f}", f"{100 * (1 - r['rmse'] / max(base_rmse, 1e-9)):.1f}%",
          f"{r['gmae']:.3f}"] for n, r in res.items()],
        ["candidate", "MAE m", "RMSE m", "detail gain", "grad MAE m/px"])
    ref = next(iter(res.values()))
    print("\n== terrain statistics: candidate / teacher  (1.00 = matches the teacher) ==")
    rows = []
    for n, r in res.items():
        rows.append([n, f"{r['s50_p'] / max(r['s50_t'], 1e-12):.2f}", f"{r['s90_p'] / max(r['s90_t'], 1e-12):.2f}",
                     f"{r['s99_p'] / max(r['s99_t'], 1e-12):.2f}", f"{r['rough_p'] / max(r['rough_t'], 1e-12):.2f}",
                     f"{r['hf_p'] / max(r['hf_t'], 1e-12):.2f}", f"{r['slope_p']:.2f}"])
    fmt(rows, ["candidate", "slope p50", "slope p90", "slope p99", "roughness", "HF energy", "spectral slope"])
    print(f"teacher spectral slope: {ref['slope_t']:.2f}; teacher slope p50/p90/p99 = "
          f"{ref['s50_t']:.3f}/{ref['s90_t']:.3f}/{ref['s99_t']:.3f} (m/m)")
    print("(HF energy < 1 and roughness < 1 mean blur / missing detail; > 1 means over-sharpened or noisy.)")


def print_seam(name, r):
    print(f"\n== seam / statelessness test: {name} ==")
    fmt([[f"{lo}-{hi}", f"{mx:.4f}", f"{mn:.4f}"] for lo, hi, mx, mn in r["table"]],
        ["dist to window edge (fine px)", "max err m", "mean err m"])
    if r["halo_fine_px"] is None:
        print(f"-> errors never drop below {r['eff_tol']:.3f} m (far-from-edge error {r['floor']:.3f} m): "
              f"output depends on the window (NOT stateless)")
    else:
        note = (f" (numerical floor ~{r['floor']:.3f} m far from edges: precision / kernel choice, not window dependence;"
                f" rerun without --tf32 to check)") if r["floor"] > r["tol"] else ""
        print(f"-> error <= {r['eff_tol']:.3f} m beyond {r['halo_fine_px']} fine px from a window edge: "
              f"use a halo of >= {r['halo_fine_px']} fine px ({int(np.ceil(r['halo_fine_px'] / 2))} coarse px)."
              f" worst error anywhere {r['worst']:.3f} m{note}")


def bench(gen, size=512, reps=5):
    import torch
    coarse = np.random.default_rng(0).normal(300, 100, (size, size)).astype(np.float32)
    dev = gen.device
    sync = (lambda: torch.cuda.synchronize()) if str(dev).startswith("cuda") else (lambda: None)
    gen(coarse, (0, 0))
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        gen(coarse, (0, 0))
    sync()
    e2e = (time.perf_counter() - t0) / reps
    cfg, model = gen.cfg, gen.model
    x = torch.randn(1, 1 + cfg["n_noise"], 2 * size, 2 * size, device=dev)
    with torch.no_grad():
        model(x)
        sync()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        sync()
    mo = (time.perf_counter() - t0) / reps
    mpx = (2 * size) ** 2 / 1e6
    print(f"   speed ({2 * size}x{2 * size} output): model-only {1000 * mo / mpx:.0f} ms/Mpx, "
          f"end-to-end {1000 * e2e / mpx:.0f} ms/Mpx (end-to-end includes CPU noise hashing + copies)")


def montage(path, coarse_fine, outs, mpp, order):
    try:
        from PIL import Image
    except ImportError:
        print("[note] Pillow not installed; skipping --png (pip install pillow)")
        return
    S = 256
    rows = []
    for si in range(len(outs[order[0]])):
        fine = coarse_fine[si][1]
        cols = [outs[n][si] for n in order] + [fine]
        h0, w0 = fine.shape
        a, b = (h0 - S) // 2, (w0 - S) // 2
        tiles = [(hillshade(c[a:a + S, b:b + S].astype(np.float64), mpp) * 255).astype(np.uint8) for c in cols]
        rows.append(np.concatenate(tiles, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(path)
    print(f"[png] wrote {path}  (columns: {', '.join(order)}, teacher)")


# =============================================================== cli


def load_pairs(a):
    man = load_manifest(a.data)
    levels = a.levels or man["levels"]
    L = a.level if a.level is not None else levels - 1
    ds = PyramidData(a.data, levels=levels, augment=False, space="metres", norm=1.0)
    n = min(a.samples, len(ds))
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    pairs = []
    for i in idx:
        it = ds[int(i)]
        pairs.append((it["levels"][L - 1][0].astype(np.float32), it["levels"][L][0].astype(np.float32)))
    return pairs, man["mpp"] * 2 ** (levels - 1 - L), L


def cmd_compare(a):
    pairs, mpp, L = load_pairs(a)
    print(f"[data] {len(pairs)} validation patches, level {L}, {mpp:.0f} m/px output")
    cands = make_candidates(a.candidates, a)
    res, outs = compare(pairs, cands, mpp)
    print_comparison(res)
    if a.seam:
        coarse = pairs[0][0]
        for n, g in cands.items():
            print_seam(n, seam_test(g, coarse, a.tol))
    if a.bench:
        print("\n== speed ==")
        for n, g in cands.items():
            if hasattr(g, "model"):
                print(f" {n}")
                bench(g)
    if a.png:
        montage(a.png, pairs, outs, mpp, list(cands))


def cmd_selftest(a):
    import shutil
    import tempfile
    from make_pyramid_data import cmd_generate
    rng = np.random.default_rng(0)
    # spectrum: white noise -> slope ~ 0 ; smoothed noise -> clearly negative
    wn = rng.normal(size=(256, 256))
    assert abs(spectrum_stats(wn)[0]) < 0.3
    sm_ = wn.copy()
    for _ in range(4):
        sm_ = (sm_ + np.roll(sm_, 1, 0) + np.roll(sm_, 1, 1) + np.roll(sm_, -1, 0) + np.roll(sm_, -1, 1)) / 5
    assert spectrum_stats(sm_)[0] < spectrum_stats(wn)[0] - 1.0

    # seam test: a local generator (3 passes of zero-padded laplacian smoothing) has a small halo,
    # a window-normalising generator has none
    def local_gen(c, origin=(0, 0)):
        z = up2(np.asarray(c, np.float32))
        for _ in range(3):
            zp = np.pad(z, 1)
            z = z + 0.1 * (zp[:-2, 1:-1] + zp[2:, 1:-1] + zp[1:-1, :-2] + zp[1:-1, 2:] - 4 * z)
        return z

    def stateful_gen(c, origin=(0, 0)):
        z = up2(np.asarray(c, np.float32))
        return z - z.mean()

    coarse = rng.normal(0, 50, (96, 128)).astype(np.float32)
    r1, r2 = seam_test(local_gen, coarse), seam_test(stateful_gen, coarse)
    print_seam("local_gen (expect small halo)", r1)
    print_seam("stateful_gen (expect NOT stateless)", r2)
    assert r1["halo_fine_px"] is not None and r1["halo_fine_px"] <= 8, r1
    assert r2["halo_fine_px"] is None, r2

    # comparison on demo data: blurred must lose HF energy, noisy must gain it, bilinear has zero gain
    d = tempfile.mkdtemp()
    try:
        ns = argparse.Namespace(out=d, repo=".", model="x", demo=True, blocks=4, block=512, patch=256, levels=4,
                                seed=0, data_seed=3, span=20000, min_std=5.0, max_ocean=0.5, with_climate=False,
                                dtype="fp32", batch="1", compile=False, cache_gb=1.0, device=None)
        cmd_generate(ns)
        a2 = argparse.Namespace(data=d, levels=4, level=3, samples=6)
        pairs, mpp, L = load_pairs(a2)
        assert L == 3 and mpp == 30.0 and pairs[0][1].shape == (256, 256) and pairs[0][0].shape == (128, 128)

        def blur(c, origin=(0, 0)):
            z = up2(c)
            zp = np.pad(z, 1, mode="edge")
            return (zp[:-2, 1:-1] + zp[2:, 1:-1] + zp[1:-1, :-2] + zp[1:-1, 2:] + z) / 5

        noisy_rng = np.random.default_rng(1)
        cands = collections.OrderedDict(bilinear=bilinear_candidate(), blur=blur,
                                        noisy=lambda c, origin=(0, 0): up2(c) + noisy_rng.normal(0, 3.0, (2 * c.shape[0], 2 * c.shape[1])))
        res, outs = compare(pairs, cands, mpp)
        print_comparison(res)
        assert res["blur"]["hf_p"] < res["bilinear"]["hf_p"] < res["noisy"]["hf_p"]
        assert abs(1 - res["bilinear"]["rmse"] / res["bilinear"]["rmse"]) < 1e-12
        if a.png:
            montage(a.png, pairs, outs, mpp, list(cands))
    finally:
        shutil.rmtree(d)
    print("SELFTEST PASSED")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compare")
    c.add_argument("--data", required=True, help="validation dataset dir (from make_pyramid_data.py generate)")
    c.add_argument("--candidates", default="bilinear")
    c.add_argument("--levels", type=int, default=None)
    c.add_argument("--level", type=int, default=None, help="output level to evaluate (default finest)")
    c.add_argument("--samples", type=int, default=32)
    c.add_argument("--seam", action="store_true")
    c.add_argument("--tol", type=float, default=0.01, help="seam tolerance in metres")
    c.add_argument("--bench", action="store_true")
    c.add_argument("--png", default=None)
    c.add_argument("--device", default=None)
    c.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"], help="student inference precision")
    c.add_argument("--noise-seed", type=int, default=0)
    c.add_argument("--tf32", action="store_true",
                   help="allow TF32 convs (faster on Ampere, but adds ~cm-level numerical noise to the seam test)")
    c.set_defaults(fn=cmd_compare)
    s = sub.add_parser("selftest")
    s.add_argument("--png", default=None)
    s.set_defaults(fn=cmd_selftest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()