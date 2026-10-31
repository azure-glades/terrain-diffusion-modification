#!/usr/bin/env python3
"""Where does Terrain Diffusion spend its time?  Run on the machine with the GPU.

    cd ~/terrain-diffusion                       # your clone (needs its requirements installed)
    python /path/to/profile_terrain.py --quick                    # ~2-5 min
    python /path/to/profile_terrain.py --dtype bf16 --ops         # + kernel-level breakdown
    python /path/to/profile_terrain.py --sweep                    # fp32 / tf32 / bf16 / fp16 / batch-size comparison
    python /path/to/profile_terrain.py --repo ~/terrain-diffusion # if not run from the clone

What it measures (all on cold, never-seen regions unless stated):
  1. first-request latency (model warm-up, cudnn autotune, torch.compile if enabled)
  2. per-stage GPU time for coarse / base(latent) / decoder via forward hooks (CUDA events, no extra syncs)
     + orchestration overhead = wall time - GPU model time (python, infinite-tensor, caching, blending)
  3. cold latency vs request size, ms per megapixel, and tile "work amplification" from window overlap
  4. warm (cached) request cost
  5. the explorer's access pattern: chunk-by-chunk walk vs 4-chunk blocks vs one long strip
  6. optional (--ops): torch.profiler op breakdown (attention / conv / norm / gemm / other)
Writes profile_results.json and prints an auto-diagnosis mapped to optimisation directions.
"""
import argparse
import contextlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    ap.add_argument("--repo", default=None, help="path to terrain-diffusion clone (added to sys.path)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    ap.add_argument("--batch", default="1", help="latents_batch_size: '4' or list '1,4,8'")
    ap.add_argument("--compile", action="store_true", help="torch_compile=True (slow first request)")
    ap.add_argument("--tf32", action="store_true", help="allow TF32 matmul/conv (Ampere+ GPUs)")
    ap.add_argument("--onestep-latent", action="store_true", help="pass onestep_latent=True to the pipeline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sizes", default="128,256,512,1024", help="square request sizes in px")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--walk-chunk", type=int, default=128)
    ap.add_argument("--walk-steps", type=int, default=24)
    ap.add_argument("--ops", action="store_true", help="torch.profiler op breakdown on one cold request")
    ap.add_argument("--quick", action="store_true", help="sizes 256,512; 2 repeats; 12 walk steps")
    ap.add_argument("--json-out", default="profile_results.json")
    ap.add_argument("--sweep", action="store_true", help="re-run this script across dtype/batch/tf32 configs")
    return ap.parse_args()


# ------------------------------------------------------------------ helpers
def med(xs):
    return statistics.median(xs) if xs else float("nan")


def fmt_table(rows, header):
    w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w[i]) if i == 0 else str(c).rjust(w[i]) for i, c in enumerate(r))
    print(line(header))
    print("  ".join("-" * x for x in w))
    for r in rows:
        print(line(r))


def parse_batch(s):
    return [int(x) for x in s.split(",")] if "," in s else int(s)


class Env:
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.cuda = device.startswith("cuda")

    def sync(self):
        if self.cuda:
            self.torch.cuda.synchronize()


class StageTimer:
    """Forward hooks on one model: CUDA-event timing of each call, plus batch/shape bookkeeping."""

    def __init__(self, env, name, module):
        self.env, self.name = env, name
        self.reset()
        try:
            module.register_forward_pre_hook(self._pre, with_kwargs=True)
        except TypeError:
            module.register_forward_pre_hook(lambda m, a: self._pre(m, a, {}))
        module.register_forward_hook(self._post)

    def reset(self):
        self.ev, self.calls, self.elems, self.shape, self.batches = [], 0, 0, None, []

    @staticmethod
    def _first_tensor(args, kwargs):
        import torch
        for v in list(args) + list(kwargs.values()):
            if isinstance(v, torch.Tensor):
                return v
        return None

    def _pre(self, module, args, kwargs=None):
        t = self._first_tensor(args, kwargs or {})
        if t is not None and t.dim() >= 3:
            b = t.shape[0]
            self.elems += b * t.shape[-1] * t.shape[-2]
            self.batches.append(b)
            self.shape = tuple(t.shape[1:])
        self.calls += 1
        if self.env.cuda:
            s = self.env.torch.cuda.Event(enable_timing=True)
            s.record()
            self.ev.append([s, None])
        else:
            self.ev.append([time.perf_counter(), None])

    def _post(self, module, args, output):
        if self.env.cuda:
            e = self.env.torch.cuda.Event(enable_timing=True)
            e.record()
            self.ev[-1][1] = e
        else:
            self.ev[-1][1] = time.perf_counter()

    def result(self):
        if self.env.cuda:
            ms = sum(s.elapsed_time(e) for s, e in self.ev if e is not None)
        else:
            ms = sum((e - s) * 1000 for s, e in self.ev if e is not None)
        return dict(calls=self.calls, ms=ms, elems=self.elems, shape=self.shape,
                    avg_batch=(sum(self.batches) / len(self.batches)) if self.batches else 0)


def make_pipe(args, env):
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    kw = dict(seed=args.seed, dtype=None if args.dtype == "fp32" else args.dtype,
              torch_compile=args.compile, latents_batch_size=parse_batch(args.batch),
              caching_strategy="direct", cache_limit=8 * 1024 ** 3, log_mode="info")
    if args.onestep_latent:
        kw["onestep_latent"] = True
    sig = inspect.signature(WorldPipeline.from_pretrained)
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        dropped = [k for k in kw if k not in sig.parameters]
        kw = {k: v for k, v in kw.items() if k in sig.parameters}
        if dropped:
            print(f"[warn] from_pretrained does not accept {dropped}; dropped")
    t0 = time.time()
    pipe = WorldPipeline.from_pretrained(args.model, **kw)
    try:
        pipe = pipe.to(env.device) or pipe
    except Exception as e:
        print(f"[warn] pipe.to({env.device}) failed: {e!r}")
    ensure_bound(pipe)
    print(f"[load] pipeline ready in {time.time() - t0:.1f}s")
    return pipe


def ensure_bound(pipe):
    """get() needs the tile stores / InfiniteTensors that WorldPipeline.bind() creates
    (symptom of skipping it: `self.residual` is None inside _compute_elev). The repo's API server
    calls bind() after loading; we try the plausible call forms until pipe.residual exists."""
    ok = lambda: getattr(pipe, "residual", None) is not None
    if ok():
        return
    attempts = [("bind()", lambda: pipe.bind()),
                ("bind(hdf5_file=None)", lambda: pipe.bind(hdf5_file=None)),
                ("bind(hdf5_file='TEMP')", lambda: pipe.bind(hdf5_file="TEMP")),
                ("__enter__()", lambda: pipe.__enter__())]
    for label, fn in attempts:
        try:
            fn()
        except Exception as e:
            print(f"[bind] {label} failed: {e!r}")
            continue
        if ok():
            print(f"[bind] pipeline bound via {label}")
            return
        print(f"[bind] {label} ran but pipe.residual is still None")
    sys.exit("Could not bind the pipeline (pipe.residual is None). Find how the repo does it:\n"
             "  grep -n 'bind(' terrain-diffusion/terrain_diffusion/inference/*.py\n"
             "and call that in ensure_bound() in profile_terrain.py (api.py's _get_pipeline() is the reference).")


def do_get(pipe, i1, j1, i2, j2):
    try:
        out = pipe.get(i1, j1, i2, j2, with_climate=False)
    except TypeError:
        out = pipe.get(i1, j1, i2, j2)
    return out


class Loc:
    """Hands out far-apart regions so cold requests never share cache with each other."""

    def __init__(self):
        self.n = 0

    def next(self):
        self.n += 1
        return self.n * 60000, -self.n * 45000


def timed(env, pipe, stages, i1, j1, i2, j2):
    for s in stages.values():
        s.reset()
    env.sync()
    t0 = time.perf_counter()
    do_get(pipe, i1, j1, i2, j2)
    env.sync()
    wall = (time.perf_counter() - t0) * 1000
    st = {k: s.result() for k, s in stages.items()}
    model = sum(v["ms"] for v in st.values())
    return dict(wall=wall, stages=st, model=model, overhead=max(wall - model, 0.0))


# ------------------------------------------------------------------ experiments
def exp_sizes(env, pipe, stages, loc, sizes, repeats):
    res = {}
    for n in sizes:
        runs = []
        for _ in range(repeats):
            i0, j0 = loc.next()
            runs.append(timed(env, pipe, stages, i0, j0, i0 + n, j0 + n))
        r = dict(wall=med([x["wall"] for x in runs]), model=med([x["model"] for x in runs]),
                 overhead=med([x["overhead"] for x in runs]), stages={})
        for k in stages:
            r["stages"][k] = dict(
                ms=med([x["stages"][k]["ms"] for x in runs]),
                calls=med([x["stages"][k]["calls"] for x in runs]),
                avg_batch=med([x["stages"][k]["avg_batch"] for x in runs]),
                elems=med([x["stages"][k]["elems"] for x in runs]),
                shape=runs[0]["stages"][k]["shape"])
        r["ms_per_mpx"] = r["wall"] / (n * n / 1e6)
        res[n] = r
    return res


def exp_warm(env, pipe, stages, loc, n=512):
    i0, j0 = loc.next()
    timed(env, pipe, stages, i0, j0, i0 + n, j0 + n)
    return timed(env, pipe, stages, i0, j0, i0 + n, j0 + n)["wall"]


def exp_walk(env, pipe, stages, loc, N, steps):
    W = N + 3                                           # same window the explorer asks for per chunk
    out = {}
    i0, j0 = loc.next()                                 # (a) chunk-by-chunk, heading east
    per = []
    for s in range(steps):
        t = timed(env, pipe, stages, i0, j0 + s * N - 1, i0 + W, j0 + s * N - 1 + W)
        per.append(t["wall"])
    out["per_chunk_first_ms"] = per[0]
    out["per_chunk_median_ms"] = med(per[1:])
    out["per_chunk_total_ms"] = sum(per)
    K = 4                                               # (b) blocks of K chunks
    i0, j0 = loc.next()
    tot = 0.0
    for s in range(0, steps, K):
        k = min(K, steps - s)
        tot += timed(env, pipe, stages, i0, j0 + s * N - 1, i0 + W, j0 + (s + k) * N + 2)["wall"]
    out["block4_total_ms"] = tot
    i0, j0 = loc.next()                                 # (c) one strip
    out["strip_total_ms"] = timed(env, pipe, stages, i0, j0 - 1, i0 + W, j0 + steps * N + 2)["wall"]
    out["steps"], out["chunk"] = steps, N
    return out


def exp_ops(env, pipe, loc, n=512):
    torch = env.torch
    from torch.profiler import ProfilerActivity, profile
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if env.cuda else [])
    i0, j0 = loc.next()
    with profile(activities=acts) as prof:
        do_get(pipe, i0, j0, i0 + n, j0 + n)
        env.sync()
    cats = dict(attention=0.0, conv=0.0, norm=0.0, gemm=0.0, other=0.0)
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "self_device_time_total", None)
        if t is None:
            t = getattr(e, "self_cuda_time_total", 0) if env.cuda else getattr(e, "self_cpu_time_total", 0)
        if not t:
            continue
        nm = e.key.lower()
        if any(s in nm for s in ("attention", "softmax", "flash", "sdpa")):
            c = "attention"
        elif "conv" in nm or "cudnn" in nm:
            c = "conv"
        elif "norm" in nm:
            c = "norm"
        elif any(s in nm for s in ("gemm", "addmm", "matmul", "linear", "bmm", "cutlass", "ampere", "sm80", "sm90")):
            c = "gemm"
        else:
            c = "other"
        cats[c] += t
        rows.append((t, e.key[:70]))
    tot = sum(cats.values()) or 1
    rows.sort(reverse=True)
    print("\n== op breakdown (one cold %dx%d request) ==" % (n, n))
    fmt_table([[k, f"{v / 1000:.1f}", f"{100 * v / tot:.0f}%"] for k, v in cats.items()], ["category", "ms", "share"])
    print("top ops:")
    for t, k in rows[:10]:
        print(f"  {t / 1000:8.1f} ms  {k}")
    return {k: v / tot for k, v in cats.items()}


# ------------------------------------------------------------------ report
def report(res, hw):
    print("\n== cold request latency vs size ==")
    rows = []
    for n, r in res["sizes"].items():
        rows.append([f"{n}x{n}", f"{r['wall']:.0f}", f"{r['model']:.0f}", f"{r['overhead']:.0f}",
                     f"{100 * r['overhead'] / max(r['wall'], 1e-9):.0f}%", f"{r['ms_per_mpx']:.0f}"])
    fmt_table(rows, ["size", "wall ms", "GPU model ms", "overhead ms", "overhead", "ms / Mpx"])

    big = max(res["sizes"])
    r = res["sizes"][big]
    print(f"\n== per-stage breakdown at {big}x{big} ==")
    rows = []
    for k, s in r["stages"].items():
        rows.append([k, int(s["calls"]), f"{s['avg_batch']:.1f}", s["shape"],
                     f"{s['ms']:.0f}", f"{100 * s['ms'] / max(r['model'], 1e-9):.0f}%"])
    fmt_table(rows, ["stage", "calls", "avg batch", "input (C,H,W)", "GPU ms", "share of model time"])

    print(f"\nfirst request (warm-up / compile): {res['first_request_ms']:.0f} ms")
    print(f"warm (cached) 512x512 request:     {res['warm_ms']:.0f} ms")
    print(f"peak GPU memory:                   {res['peak_mem_mb']:.0f} MB" if hw["cuda"] else "")

    w = res["walk"]
    n = w["steps"]
    print(f"\n== explorer access pattern: {n} chunks of {w['chunk']}px walking east ==")
    fmt_table([
        ["1 request per chunk", f"{w['per_chunk_total_ms']:.0f}", f"{w['per_chunk_total_ms'] / n:.0f}"],
        ["blocks of 4 chunks", f"{w['block4_total_ms']:.0f}", f"{w['block4_total_ms'] / n:.0f}"],
        ["one long strip", f"{w['strip_total_ms']:.0f}", f"{w['strip_total_ms'] / n:.0f}"],
    ], ["strategy", "total ms", "ms / chunk"])
    print(f"per-chunk: first {w['per_chunk_first_ms']:.0f} ms, median of rest {w['per_chunk_median_ms']:.0f} ms")


def diagnose(res, args):
    big = max(res["sizes"])
    r = res["sizes"][big]
    out = []
    model = max(r["model"], 1e-9)
    share = {k: s["ms"] / model for k, s in r["stages"].items()}
    top = max(share, key=share.get)
    oh = r["overhead"] / max(r["wall"], 1e-9)
    out.append(f"Biggest GPU stage: {top} ({100 * share[top]:.0f}% of model time).")
    if top.startswith("decoder"):
        out.append("  -> Direction 6 (lighter decoder / distil decoder) and decoder_tile_stride are the levers.")
    elif top.startswith("base"):
        out.append("  -> Latent stage dominates: fewer/lighter steps, onestep_latent, smaller or int8 network, bigger batches.")
    elif top.startswith("coarse"):
        out.append("  -> Coarse stage dominates (20 solver steps). It is cached per large tile, so check LOD/caching first.")
    if oh > 0.30:
        out.append(f"Orchestration overhead is {100 * oh:.0f}% of wall time (GPU is idle): batch more windows per call, "
                   "use larger window-aligned requests, check caching_strategy, reduce Python/HDF5 overhead.")
    else:
        out.append(f"Overhead is only {100 * oh:.0f}% of wall time: the GPU is the bottleneck, so model-level changes pay off.")
    for k, s in r["stages"].items():
        if s["avg_batch"] and s["avg_batch"] < 2 and s["calls"] > 4 and "decoder" not in k:
            out.append(f"{k}: average batch {s['avg_batch']:.1f} over {int(s['calls'])} calls -> try --batch 1,4,8 (GPU underfed).")
    w = res["walk"]
    pc, st = w["per_chunk_total_ms"], w["strip_total_ms"]
    if st > 0 and pc / st > 1.8:
        out.append(f"Chunk-by-chunk requests cost {pc / st:.1f}x one long strip -> request larger window-aligned blocks "
                   "(the explorer's --chunk and prefetch pattern should batch neighbours).")
    if args.dtype == "fp32":
        out.append("Running fp32: try --dtype bf16 (or --tf32) -- usually the cheapest 1.5-3x.")
    if not args.compile:
        out.append("torch.compile is off: try --compile (Linux; first request will be slow).")
    return out


# ------------------------------------------------------------------ main
def run_single(args):
    args.json_out = os.path.abspath(args.json_out)           # keep output where you launched from
    if args.repo:
        repo = os.path.abspath(os.path.expanduser(args.repo))
        if not os.path.isdir(os.path.join(repo, "terrain_diffusion")):
            sys.exit(f"--repo {repo} does not look like the terrain-diffusion clone (no terrain_diffusion/ inside)")
        # the repo loads data/global/*.tif and checkpoints/ via paths relative to its root
        os.chdir(repo)
        sys.path.insert(0, repo)
    sys.path.insert(0, os.getcwd())
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = Env(torch, device)
    if env.cuda and args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.quick:
        args.sizes, args.repeats, args.walk_steps = "256,512", 2, 12
    sizes = [int(s) for s in args.sizes.split(",")]
    if not env.cuda:
        print("[warn] no CUDA device: timings are CPU-only; reducing sizes")
        sizes = [s for s in sizes if s <= 256] or [128]
    hw = dict(cuda=env.cuda, device=device, torch=torch.__version__,
              gpu=torch.cuda.get_device_name(0) if env.cuda else "cpu")
    print(f"[env] {hw}  dtype={args.dtype} batch={args.batch} compile={args.compile} tf32={args.tf32}")

    pipe = make_pipe(args, env)
    stages = {}
    for attr in ("coarse_model", "base_model", "decoder_model"):
        m = getattr(pipe, attr, None)
        if m is None:
            print(f"[warn] pipeline has no attribute {attr}; not timed")
            continue
        stages[attr.replace("_model", "")] = StageTimer(env, attr, m)
    if not stages:
        sys.exit("could not find coarse_model/base_model/decoder_model on the pipeline")
    if env.cuda:
        torch.cuda.reset_peak_memory_stats()

    loc = Loc()
    i0, j0 = loc.next()
    first = timed(env, pipe, stages, i0, j0, i0 + 256, j0 + 256)["wall"]
    print(f"[run] first request done ({first:.0f} ms); measuring...")
    res = dict(config=dict(dtype=args.dtype, batch=args.batch, compile=args.compile, tf32=args.tf32,
                           onestep_latent=args.onestep_latent),
               hw=hw, first_request_ms=first)
    res["sizes"] = exp_sizes(env, pipe, stages, loc, sizes, args.repeats)
    res["warm_ms"] = exp_warm(env, pipe, stages, loc, 256 if not env.cuda else 512)
    res["walk"] = exp_walk(env, pipe, stages, loc, args.walk_chunk, args.walk_steps)
    res["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20 if env.cuda else 0
    if args.ops:
        try:
            res["ops"] = exp_ops(env, pipe, loc)
        except Exception as e:
            print(f"[warn] op profiling failed: {e!r}")
    report(res, hw)
    print("\n== diagnosis ==")
    res["diagnosis"] = diagnose(res, args)
    for line in res["diagnosis"]:
        print(" *", line)
    with open(args.json_out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"\n[done] wrote {args.json_out}")
    return res


SWEEP = [("fp32", "1", False), ("fp32", "1", True), ("bf16", "1", False), ("fp16", "1", False),
         ("bf16", "1,4,8", False)]


def run_sweep(args):
    rows = []
    for dtype, batch, tf32 in SWEEP:
        tmp = tempfile.mktemp(suffix=".json")
        cmd = [sys.executable, os.path.abspath(__file__), "--quick", "--dtype", dtype, "--batch", batch,
               "--json-out", tmp, "--model", args.model, "--seed", str(args.seed)]
        if tf32:
            cmd.append("--tf32")
        if args.compile:
            cmd.append("--compile")
        if args.repo:
            cmd += ["--repo", args.repo]
        if args.device:
            cmd += ["--device", args.device]
        label = f"{dtype}{'+tf32' if tf32 else ''} batch={batch}"
        print(f"\n#### {label}")
        rc = subprocess.call(cmd)
        if rc != 0 or not os.path.exists(tmp):
            rows.append([label, "FAILED", "", "", "", ""])
            continue
        r = json.load(open(tmp))
        os.remove(tmp)
        s = r["sizes"][str(max(int(k) for k in r["sizes"]))]
        rows.append([label, f"{r['first_request_ms']:.0f}", f"{s['wall']:.0f}", f"{s['ms_per_mpx']:.0f}",
                     f"{r['walk']['per_chunk_total_ms'] / r['walk']['steps']:.0f}", f"{r['peak_mem_mb']:.0f}"])
    print("\n================ sweep summary ================")
    fmt_table(rows, ["config", "first req ms", "cold wall ms (largest)", "ms / Mpx", "ms / chunk (walk)", "peak MB"])


if __name__ == "__main__":
    a = parse()
    run_sweep(a) if a.sweep else run_single(a)