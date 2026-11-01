"""Is the student cascade really faster than the teacher? Same machine, same request sizes, cold regions.

Teacher : the repo's WorldPipeline, cold request of N x N px at far-apart locations (everything it does: coarse, latent, decoder).
Student : levels 1..4 chained on a level-0 window just large enough for a clean N x N output (halo included -- that is the
          real cost of a stateless request). Level 0 is NOT included: a procedural level 0 is cheap, a learned one adds to this.

  python bench_cascade.py --repo terrain-diffusion/ \
      --students runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt --sizes 256,512,1024

Use --skip-teacher to time only the students (e.g. to re-run after retraining).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

from eval_terrain import student_candidate

try:
    import torch
except ImportError:                       # lets the student path be exercised without torch (stubs)
    torch = None


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None, help="terrain-diffusion clone (needed unless --skip-teacher)")
    ap.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    ap.add_argument("--students", required=True, help="comma-separated checkpoints for levels 1..N")
    ap.add_argument("--sizes", default="256,512,1024", help="square output sizes in finest-level px")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--teacher-dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch", default="1,4,8,16", help="teacher latents_batch_size list")
    ap.add_argument("--student-dtypes", default="fp32,bf16", help="student precisions to time")
    ap.add_argument("--walk-chunk", type=int, default=128, help="explorer chunk size for the per-chunk comparison")
    ap.add_argument("--walk-steps", type=int, default=12)
    ap.add_argument("--skip-teacher", action="store_true")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="teacher world seed")
    ap.add_argument("--device", default=None)
    ap.add_argument("--json-out", default="bench_cascade.json")
    return ap.parse_args()


def sync():
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def med(xs):
    return float(np.median(xs))


def fmt(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    print(line(header))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))


# ------------------------------------------------------------------ teacher
def bench_teacher(a, sizes):
    sys.path.insert(0, os.getcwd())
    import profile_terrain as pt
    ns = argparse.Namespace(model=a.model, seed=a.seed, dtype=a.teacher_dtype, compile=False, batch=a.batch,
                            onestep_latent=False)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = pt.Env(torch, dev)
    pipe = pt.make_pipe(ns, env)
    loc = pt.Loc()

    def timed(i1, j1, i2, j2):
        env.sync()
        t0 = time.perf_counter()
        pt.do_get(pipe, i1, j1, i2, j2)
        env.sync()
        return (time.perf_counter() - t0) * 1000

    i0, j0 = loc.next()
    print("[teacher] warm-up request (not counted)")
    timed(i0, j0, i0 + 256, j0 + 256)
    res = {"cold": {}, "walk": {}}
    for n in sizes:
        ts = []
        for _ in range(a.repeats):
            i0, j0 = loc.next()
            ts.append(timed(i0, j0, i0 + n, j0 + n))
        res["cold"][n] = med(ts)
        print(f"[teacher] {n}x{n}: {res['cold'][n]:.0f} ms  ({res['cold'][n] / (n * n / 1e6):.0f} ms/Mpx)", flush=True)
    N, W = a.walk_chunk, a.walk_chunk + 3                 # same window the explorer asks for per chunk
    i0, j0 = loc.next()
    per = [timed(i0, j0 + s * N - 1, i0 + W, j0 + s * N - 1 + W) for s in range(a.walk_steps)]
    res["walk"] = dict(first_ms=per[0], median_ms=med(per[1:]), chunk=N)
    print(f"[teacher] walking chunk-by-chunk ({N}px): first {per[0]:.0f} ms, then {med(per[1:]):.0f} ms median", flush=True)
    pipe = None
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


# ------------------------------------------------------------------ student
def smooth_terrain(n, seed):
    """Plausible-looking level-0 content (timing does not depend on it, but keep values sane)."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)[:, None] ** 2 + np.fft.fftfreq(n)[None, :] ** 2
    spec = np.fft.fft2(rng.standard_normal((n, n))) / np.maximum(f, 1e-4) ** 0.9
    z = np.real(np.fft.ifft2(spec))
    return (800 * z / max(z.std(), 1e-9)).astype(np.float32)


def plan(gens, N):
    levels = sorted(gens)
    margin = 0
    for L in levels:
        margin = 2 * margin + int(gens[L].cfg.get("rf", 32))
    total = gens[levels[0]].cfg["levels"]
    f = 2 ** (total - 1)
    h0 = math.ceil(margin / f)
    n0 = math.ceil(N / f) + 2 * h0
    return levels, f, h0, n0, margin


def run_cascade(gens, levels, x0, oi, oj, per_level):
    z = x0
    for L in levels:
        t0 = time.perf_counter()
        z = gens[L](z, (oi * 2 ** (L - 1), oj * 2 ** (L - 1)))
        per_level.setdefault(L, []).append((time.perf_counter() - t0) * 1000)
    return z


def bench_student(a, sizes, dtype):
    gens = {}
    for p in a.students.split(","):
        g = student_candidate(os.path.abspath(p.strip()) if os.path.exists(p.strip()) else p.strip(),
                              a.device, a.noise_seed, dtype)
        gens[g.cfg["level"]] = g
    out = {}
    for N in sizes:
        levels, f, h0, n0, margin = plan(gens, N)
        x0 = smooth_terrain(n0, 1)
        computed = sum((n0 * 2 ** L) ** 2 for L in levels) / 1e6
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            run_cascade(gens, levels, x0, 0, 0, {})                       # warm-up (cudnn autotune etc.)
            ts, per = [], {}
            for r in range(a.repeats):
                sync()
                t0 = time.perf_counter()
                run_cascade(gens, levels, x0, 1000 * (r + 1), -700 * (r + 1), per)
                ts.append((time.perf_counter() - t0) * 1000)
            peak = torch.cuda.max_memory_allocated() / 2 ** 20 if (torch is not None and torch.cuda.is_available()) else 0.0
            out[N] = dict(ms=med(ts), per_level={L: med(v) for L, v in per.items()}, level0_px=n0, halo0=h0,
                          out_px=n0 * f, computed_mpx=computed, peak_mb=peak)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            out[N] = dict(oom=True, level0_px=n0, out_px=n0 * f)
            if torch is not None:
                torch.cuda.empty_cache()
        r = out[N]
        print(f"[student {dtype or 'fp32'}] {N}x{N}: " + ("OOM" if r.get("oom") else f"{r['ms']:.0f} ms") +
              f"  (level-0 window {r['level0_px']}px -> {r['out_px']}px computed at the finest level)", flush=True)
    # per-chunk cost in the explorer: every chunk is its own stateless request
    Nc, W = a.walk_chunk, a.walk_chunk + 3
    levels, f, h0, n0, margin = plan(gens, W)
    x0 = smooth_terrain(n0, 2)
    run_cascade(gens, levels, x0, 0, 0, {})
    ts = []
    for r in range(max(a.repeats, 3)):
        sync()
        t0 = time.perf_counter()
        run_cascade(gens, levels, x0, 50 * (r + 1), 30 * (r + 1), {})
        ts.append((time.perf_counter() - t0) * 1000)
    walk = dict(chunk_ms=med(ts), chunk=Nc, level0_px=n0)
    return out, walk, dict(margin_fine_px=margin)


def main():
    a = parse()
    sizes = [int(s) for s in a.sizes.split(",")]
    a.json_out = os.path.abspath(a.json_out)
    a.students = ",".join(os.path.abspath(p.strip()) for p in a.students.split(","))
    results = {}
    if not a.skip_teacher:
        if not a.repo:
            sys.exit("--repo is required for the teacher (or use --skip-teacher)")
        repo = os.path.abspath(os.path.expanduser(a.repo))
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        os.chdir(repo)
        sys.path.insert(0, repo)
        results["teacher"] = bench_teacher(a, sizes)
    results["student"] = {}
    for d in a.student_dtypes.split(","):
        d = d.strip()
        results["student"][d] = bench_student(a, sizes, "bf16" if d == "bf16" else None)

    print("\n================ cold request: ms per request and per Mpx of useful output ================")
    header = ["size", "teacher ms", "teacher ms/Mpx"]
    for d in results["student"]:
        header += [f"student {d} ms", f"{d} ms/Mpx", f"{d} speedup"]
    header += ["compute overhead"]
    rows = []
    for N in sizes:
        mpx = N * N / 1e6
        t = results.get("teacher", {}).get("cold", {}).get(N)
        row = [f"{N}x{N}", f"{t:.0f}" if t else "-", f"{t / mpx:.0f}" if t else "-"]
        ov = ""
        for d, (res, walk, info) in results["student"].items():
            r = res[N]
            if r.get("oom"):
                row += ["OOM", "-", "-"]
                continue
            row += [f"{r['ms']:.0f}", f"{r['ms'] / mpx:.0f}", f"{t / r['ms']:.1f}x" if t else "-"]
            ov = f"{r['computed_mpx'] / mpx:.1f}x pixels"
        rows.append(row + [ov])
    fmt(rows, header)
    first = next(iter(results["student"].values()))
    big = max((n for n in sizes if not first[0][n].get("oom")), default=None)
    if big:
        print(f"\nper-level time of the student cascade at {big}x{big} (ms): " +
              "  ".join(f"L{L} {v:.0f}" for L, v in first[0][big]["per_level"].items()))
        print(f"student peak GPU memory at {big}x{big}: {first[0][big]['peak_mb']:.0f} MiB;  "
              f"halo needed: {first[2]['margin_fine_px']} finest px each side ({first[0][big]['halo0']} level-0 px)")

    print(f"\n================ explorer-style chunks ({a.walk_chunk}px, each its own request) ================")
    wrows = []
    tw = results.get("teacher", {}).get("walk")
    if tw:
        wrows.append(["teacher (cache warm after first)", f"{tw['median_ms']:.0f}", f"first chunk {tw['first_ms']:.0f}"])
    for d, (res, walk, info) in results["student"].items():
        wrows.append([f"student {d}, stateless chunk", f"{walk['chunk_ms']:.0f}",
                      f"{walk['level0_px']}px level-0 window"])
        if 512 in res and not res[512].get("oom"):
            per = res[512]["ms"] / ((512 / a.walk_chunk) ** 2)
            wrows.append([f"student {d}, amortised (512 region cut into chunks)", f"{per:.0f}", ""])
    fmt(wrows, ["case", "ms per chunk", "note"])
    print("\nNotes:")
    print(" - The student excludes level 0 (the coarse continent-scale layer). The teacher time includes all of its stages.")
    print(" - 'compute overhead' = pixels the cascade actually computes / useful output pixels: the halo (context) cost of statelessness.")
    print(" - Small chunks pay the halo on every request; generating larger regions once and cutting chunks from them amortises it.")
    print(" - The teacher keeps a tile cache, so its chunk-by-chunk cost drops after the first chunk; the student has no cache here.")
    with open(a.json_out, "w") as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"[json] {a.json_out}")


if __name__ == "__main__":
    main()