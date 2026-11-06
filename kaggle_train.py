"""Train on a Kaggle notebook (GPU session): the dedicated GAN students (l1_gan, l2_gan) and/or the SHARED student.

    !python -u kaggle_train.py --only shared                 # one network for levels 1-4, warm-started from l1_gan
    !python -u kaggle_train.py                                # l1_gan + l2_gan (dedicated students), as before
    !python -u kaggle_train.py --only l1_gan,l2_gan,shared --hours 11
    !python -u kaggle_train.py --repo /kaggle/working/terrain_explorer_milestone --train data/train2,data/train3

What it does
  * finds the repo (the folder containing train_level.py / train_shared.py), the data dirs and runs/
  * data: by default every one of data/train2, data/train3 and data/val2, data/val3 that exists is used (comma lists)
  * 2 GPUs -> jobs are spread over cuda:0 / cuda:1;  1 GPU -> all jobs share it in parallel (--sequential splits the time)
  * every job uses --resume and --max-minutes: when the session's time is up the state is saved, and running the same
    cell in the next session continues from it. Total step counts stay the same across sessions.
  * writes into /kaggle/working/runs (input datasets are read-only, so existing runs are copied there first)
  * at the end, zips the finished runs into /kaggle/working/runs_<jobs>.zip so they survive the session
Shared job: --shared-levels (default 1,2,3,4), --shared-init (default: runs/l2_gan/last.pt if it exists, else scratch;
'none' to force scratch, or a checkpoint path), --steps-shared, and --shared-args "<extra train_shared.py flags>" for anything else.
Kaggle limits: ~12 h per GPU session, /kaggle/working about 20 GB, 4 CPU cores, ~29 GB RAM. Turn the GPU on in
Settings > Accelerator.
"""
import argparse
import glob
import os
import shlex
import shutil
import subprocess
import sys
import time

SCRIPT = {"l1_gan": "train_level.py", "l2_gan": "train_level.py", "shared": "train_shared.py"}


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
        hits = [h for h in glob.glob(os.path.join(root, "**", rel), recursive=True) if os.path.isdir(h)]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def resolve_list(repo, spec, defaults, what):
    """spec: comma list given by the user (each item a path or a name under the repo), or None -> defaults that exist."""
    items = [s.strip() for s in spec.split(",") if s.strip()] if spec else list(defaults)
    out = []
    for it in items:
        d = it if os.path.isdir(it) else find_dir(repo, it)
        if d:
            out.append(os.path.abspath(d))
        elif spec:
            sys.exit(f"{what} directory '{it}' not found")
    if not out:
        sys.exit(f"no {what} data found (looked for {list(defaults)}); pass --{what}")
    return ",".join(out)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--train", default=None, help="comma list of training dirs (default: data/train2,data/train3 if present)")
    ap.add_argument("--val", default=None, help="comma list of validation dirs (default: data/val2,data/val3 if present)")
    ap.add_argument("--runs", default=None, help="existing runs dir to resume from (default: <repo>/runs if present)")
    ap.add_argument("--out", default="/kaggle/working/runs", help="where runs are written (must be writable)")
    ap.add_argument("--hours", type=float, default=11.0, help="training time budget for this session")
    ap.add_argument("--only", default="l1_gan,l2_gan", help="comma list of: l1_gan, l2_gan, shared")
    ap.add_argument("--steps-l1", type=int, default=30000)
    ap.add_argument("--steps-l2", type=int, default=40000)
    ap.add_argument("--steps-shared", type=int, default=80000)
    ap.add_argument("--shared-levels", default="1,2,3,4")
    ap.add_argument("--shared-init", default="auto",
                    help="'auto' (= runs/l2_gan/last.pt), 'none', or a checkpoint path; used only on the first start")
    ap.add_argument("--shared-args", default="", help="extra flags for train_shared.py, e.g. '--weights 1:2,2:2,3:1,4:1'")
    ap.add_argument("--workers", type=int, default=0, help="data-loader workers per job (0 = auto: 4 for one job, else 2)")
    ap.add_argument("--sequential", action="store_true", help="one GPU: run the jobs one after the other (time split)")
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
    names = [n.strip() for n in a.only.split(",") if n.strip()]
    unknown = [n for n in names if n not in SCRIPT]
    if unknown:
        sys.exit(f"unknown job(s) {unknown}; choose from {list(SCRIPT)}")
    repo = os.path.abspath(a.repo) if a.repo else find_repo()
    if "shared" in names and not os.path.exists(os.path.join(repo, "train_shared.py")):
        sys.exit(f"train_shared.py not found in {repo}: copy the latest files into the repo")
    train = resolve_list(repo, a.train, ["data/train2", "data/train3"], "train")
    val = resolve_list(repo, a.val, ["data/val2", "data/val3"], "val")
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    src_runs = a.runs or find_dir(repo, "runs")
    if src_runs and os.path.abspath(src_runs) != out:                    # copy previous state into the writable place
        for n in names:
            s, d = os.path.join(src_runs, n), os.path.join(out, n)
            if os.path.isdir(s) and not os.path.exists(d):
                print(f"[kaggle] copying {s} -> {d}")
                shutil.copytree(s, d)

    shared_init = None
    if "shared" in names and a.shared_init != "none":
        if a.shared_init != "auto":
            cands = [os.path.abspath(a.shared_init), os.path.join(repo, a.shared_init)]
        else:
            cands = [os.path.join(out, "l2_gan", "last.pt")] + \
                ([os.path.join(src_runs, "l2_gan", "last.pt")] if src_runs else [])
        shared_init = next((os.path.abspath(c) for c in cands if os.path.exists(c)), None)
        if shared_init is None:
            print("[kaggle] no l2_gan checkpoint found for --shared-init: the shared model starts from scratch "
                  "(put runs/l2_gan/last.pt in the repo's runs folder, or pass --shared-init <path>, to warm-start it)")
        else:
            print(f"[kaggle] shared model warm-starts from {shared_init}")

    import torch
    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        sys.exit("no GPU visible: enable it in Settings > Accelerator")
    print(f"[kaggle] repo {repo}\n[kaggle] train {train}\n[kaggle] val {val}\n[kaggle] runs -> {out}\n"
          f"[kaggle] {ngpu} GPU(s): " + ", ".join(torch.cuda.get_device_name(i) for i in range(ngpu)))

    gan_common = ["--n-noise", "2", "--spec-w", "0.1", "--slope-w", "0.2", "--grad-w", "0.2", "--adv-w", "0.2",
                  "--snapshot-every", "5000"]
    jobs = {
        "l1_gan": ["--level", "1", "--crop", "2048", "--bs", "4", "--depth", "4", "--steps", str(a.steps_l1),
                   "--adv-start", "3000"] + gan_common,
        "l2_gan": ["--level", "2", "--crop", "1024", "--bs", "8", "--depth", "4", "--steps", str(a.steps_l2),
                   "--adv-start", "5000"] + gan_common,
        "shared": ["--train-levels", a.shared_levels, "--noise-levels", "1,2", "--adv-levels", "1,2",
                   "--steps", str(a.steps_shared), "--adv-start", "2000" if shared_init else "5000",
                   "--snapshot-every", "5000"]
                  + (["--init", shared_init] if shared_init else []) + shlex.split(a.shared_args),
    }

    sequential = a.sequential and ngpu == 1 and len(names) > 1
    minutes = a.hours * 60 / (len(names) if sequential else 1)
    workers = a.workers or (4 if len(names) == 1 or sequential else 2)

    def launch(i, n):
        dev = f"cuda:{i % ngpu}"
        log = os.path.join(out, f"{n}.log")
        cmd = [sys.executable, "-u", SCRIPT[n], "--train", train, "--val", val, "--out", os.path.join(out, n),
               "--resume", "--no-bf16", "--workers", str(workers), "--device", dev,
               "--max-minutes", f"{minutes:.1f}"] + jobs[n]
        print(f"[kaggle] {n} ({SCRIPT[n]}) on {dev} for {minutes:.0f} min: log {log}")
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
        base = "/kaggle/working/runs_" + "_".join(names)
        stage = "/kaggle/working/_zip_stage"
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        for n in names:
            if os.path.isdir(os.path.join(out, n)):
                shutil.copytree(os.path.join(out, n), os.path.join(stage, n))
        shutil.make_archive(base, "zip", stage)
        shutil.rmtree(stage, ignore_errors=True)
        print(f"[kaggle] wrote {base}.zip ({os.path.getsize(base + '.zip') / 2 ** 20:.0f} MiB). "
              "Use 'Save Version' (or download from the Output tab) to keep it; to continue in the next session put the "
              "zip contents in <repo>/runs (or pass --runs) and run the same cell again.")


if __name__ == "__main__":
    main()