"""Train l1_gan and l2_gan on a Kaggle notebook (GPU session). Run from a cell:

    !python -u kaggle_train.py                      # auto-detects repo/data/runs under /kaggle
    !python -u kaggle_train.py --repo /kaggle/working/terrain_explorer_milestone --hours 11

What it does
  * finds the repo (the folder containing train_level.py), data/train2, data/val2 and runs/
  * 2 GPUs -> l1_gan on cuda:0 and l2_gan on cuda:1;  1 GPU -> both on it in parallel (--sequential to split the time)
  * every run uses --resume and --max-minutes, so when the session's time is up the state is saved and the next
    session continues from it (run the same cell again). Total step counts stay the same across sessions.
  * writes into /kaggle/working/runs (input datasets are read-only, so existing runs are copied there first)
  * at the end, zips runs/l1_gan and runs/l2_gan into /kaggle/working/runs_l1_l2.zip so they survive the session
Kaggle limits: ~12 h per GPU session, /kaggle/working about 20 GB, 4 CPU cores. Turn GPU on in Settings > Accelerator.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import time


def find_repo():
    for root in ("/kaggle/working", "/kaggle/input"):
        hits = glob.glob(os.path.join(root, "**", "train_level.py"), recursive=True)
        if hits:
            return os.path.dirname(sorted(hits, key=len)[0])
    sys.exit("could not find train_level.py under /kaggle; pass --repo")


def find_dir(repo, rel, extra_roots=("/kaggle/working", "/kaggle/input")):
    p = os.path.join(repo, rel)
    if os.path.isdir(p):
        return p
    for root in extra_roots:
        hits = glob.glob(os.path.join(root, "**", rel), recursive=True)
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--train", default=None, help="default: <repo>/data/train2")
    ap.add_argument("--val", default=None, help="default: <repo>/data/val2")
    ap.add_argument("--runs", default=None, help="existing runs dir to resume from (default: <repo>/runs if present)")
    ap.add_argument("--out", default="/kaggle/working/runs", help="where runs are written (must be writable)")
    ap.add_argument("--hours", type=float, default=11.0, help="training time budget for this session")
    ap.add_argument("--steps-l1", type=int, default=30000)
    ap.add_argument("--steps-l2", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers per job (Kaggle has 4 cores)")
    ap.add_argument("--sequential", action="store_true", help="one GPU: run l1_gan then l2_gan (half the time each)")
    ap.add_argument("--only", default="l1_gan,l2_gan")
    ap.add_argument("--no-zip", action="store_true")
    return ap.parse_args()


def last_line(path):
    try:
        with open(path, "rb") as f:
            f.seek(max(os.path.getsize(path) - 4000, 0))
            lines = [l for l in f.read().decode(errors="ignore").replace("\r", "\n").split("\n") if l.strip()]
        return lines[-1] if lines else ""
    except OSError:
        return ""


def main():
    a = parse()
    repo = os.path.abspath(a.repo) if a.repo else find_repo()
    train = a.train or find_dir(repo, "data/train2") or sys.exit("data/train2 not found; pass --train")
    val = a.val or find_dir(repo, "data/val2") or sys.exit("data/val2 not found; pass --val")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    names = [n.strip() for n in a.only.split(",") if n.strip()]

    src_runs = a.runs or find_dir(repo, "runs")
    if src_runs and os.path.abspath(src_runs) != out:                    # copy previous state into the writable place
        for n in names:
            s, d = os.path.join(src_runs, n), os.path.join(out, n)
            if os.path.isdir(s) and not os.path.exists(d):
                print(f"[kaggle] copying {s} -> {d}")
                shutil.copytree(s, d)

    import torch
    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        sys.exit("no GPU visible: enable it in Settings > Accelerator")
    print(f"[kaggle] repo {repo}\n[kaggle] train {train}\n[kaggle] val {val}\n[kaggle] runs -> {out}\n"
          f"[kaggle] {ngpu} GPU(s): " + ", ".join(torch.cuda.get_device_name(i) for i in range(ngpu)))

    jobs = {
        "l1_gan": ["--level", "1", "--crop", "2048", "--bs", "4", "--depth", "4", "--steps", str(a.steps_l1),
                   "--n-noise", "2", "--spec-w", "0.1", "--slope-w", "0.2", "--grad-w", "0.2",
                   "--adv-w", "0.2", "--adv-start", "3000", "--snapshot-every", "5000"],
        "l2_gan": ["--level", "2", "--crop", "1024", "--bs", "8", "--depth", "4", "--steps", str(a.steps_l2),
                   "--n-noise", "2", "--spec-w", "0.1", "--slope-w", "0.2", "--grad-w", "0.2",
                   "--adv-w", "0.2", "--adv-start", "5000", "--snapshot-every", "5000"],
    }
    unknown = [n for n in names if n not in jobs]
    if unknown:
        sys.exit(f"unknown job(s) {unknown}; choose from {list(jobs)}")

    sequential = a.sequential and ngpu == 1 and len(names) > 1
    minutes = a.hours * 60 / (len(names) if sequential else 1)

    def launch(i, n):
        dev = f"cuda:{i % ngpu}"
        log = os.path.join(out, f"{n}.log")
        cmd = [sys.executable, "-u", "train_level.py", "--train", train, "--val", val, "--out", os.path.join(out, n),
               "--resume", "--no-bf16", "--workers", str(a.workers), "--device", dev,
               "--max-minutes", f"{minutes:.1f}"] + jobs[n]
        print(f"[kaggle] {n} on {dev} for {minutes:.0f} min: log {log}")
        return subprocess.Popen(cmd, cwd=repo, stdout=open(log, "a"), stderr=subprocess.STDOUT), log

    procs = {}
    t_start = time.time()
    try:
        if sequential:
            for i, n in enumerate(names):
                p, log = launch(i, n)
                procs[n] = (p, log)
                while p.poll() is None:
                    time.sleep(60)
                    print(f"[{(time.time() - t_start) / 60:6.1f} min] {n}: {last_line(log)}", flush=True)
        else:
            for i, n in enumerate(names):
                procs[n] = launch(i, n)
            tick = 0
            while any(p.poll() is None for p, _ in procs.values()):
                time.sleep(30)
                tick += 1
                if tick % 10 == 0:                                       # every ~5 min
                    for n, (p, log) in procs.items():
                        print(f"[{(time.time() - t_start) / 60:6.1f} min] {n}: {last_line(log)}", flush=True)
    except KeyboardInterrupt:
        print("[kaggle] interrupted; stopping jobs (state is saved every 2000 steps; rerun to resume)")
        for p, _ in procs.values():
            p.terminate()
    for n, (p, log) in procs.items():
        print(f"[kaggle] {n} exit code {p.poll()}  |  {last_line(log)}")

    if not a.no_zip:
        base = "/kaggle/working/runs_l1_l2"
        stage = "/kaggle/working/_zip_stage"
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        for n in names:
            if os.path.isdir(os.path.join(out, n)):
                shutil.copytree(os.path.join(out, n), os.path.join(stage, n))
        shutil.make_archive(base, "zip", stage)
        shutil.rmtree(stage, ignore_errors=True)
        print(f"[kaggle] wrote {base}.zip ({os.path.getsize(base + '.zip') / 2 ** 20:.0f} MiB). "
              "Use 'Save Version' (or download from the Output tab) to keep it; rerun this cell in the next session "
              "to resume (put the zip contents in <repo>/runs or pass --runs).")


if __name__ == "__main__":
    main()