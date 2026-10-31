"""Terrain Diffusion explorer.

  python main.py --demo                                   # no model, procedural stand-in
  python main.py --spawn-server --repo ~/terrain-diffusion --seed 42    # starts the model server for you
  python main.py --seed 42                                 # connect to a server you started yourself:
      python -m terrain_diffusion api xandergos/terrain-diffusion-30m --seed 42
"""
import argparse
import atexit
import os
import shlex
import subprocess
import sys

from panda3d.core import loadPrcFileData


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use procedural stand-in source (no GPU/model)")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="terrain API server URL")
    ap.add_argument("--spawn-server", action="store_true", help="launch `terrain_diffusion api` as a subprocess")
    ap.add_argument("--repo", default=".", help="path to your terrain-diffusion clone (for --spawn-server)")
    ap.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    ap.add_argument("--server-args", default="", help='extra args for the server, e.g. "--dtype bf16"')
    ap.add_argument("--compile", action="store_true", help="let the server use torch.compile (default: off)")
    ap.add_argument("--onestep-latent", action="store_true",
                    help="one-step latent sampling (faster, lower quality). With --spawn-server it is passed to the "
                         "server as `--kwarg onestep_latent=True`; with your own server, start it that way yourself. "
                         "Chunks are cached separately from the normal mode, so you can A/B on the same seed.")
    ap.add_argument("--seed", type=int, default=0, help="world seed (must match the server's --seed)")
    ap.add_argument("--chunk", type=int, default=128, help="chunk size in heightmap pixels")
    ap.add_argument("--radius", type=float, default=6, help="drawn radius in chunks")
    ap.add_argument("--prefetch", type=float, default=2, help="extra generated-but-not-drawn ring, in chunks")
    ap.add_argument("--evict", type=float, default=2, help="extra margin before heights leave RAM, in chunks")
    ap.add_argument("--pregen", type=float, default=None, help="start-up pre-generation radius (default: --radius)")
    ap.add_argument("--workers", type=int, default=2, help="concurrent generation requests")
    ap.add_argument("--mpp", type=float, default=30.0, help="metres per heightmap pixel (30 for the 30m model)")
    ap.add_argument("--vexag", type=float, default=1.5, help="vertical exaggeration")
    ap.add_argument("--speed", type=float, default=60.0, help="fly speed, pixels/s")
    ap.add_argument("--start", default="0,0", help="start position 'row,col' in pixels")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--selftest", type=int, default=0, help=argparse.SUPPRESS)  # render N frames, screenshot, exit
    return ap.parse_args()


def main():
    a = parse()
    from chunks import ChunkManager
    from sources import DemoSource, HttpSource

    proc = None
    if a.demo:
        src = DemoSource(a.seed)
    else:
        src = HttpSource(a.url, seed=a.seed, model=os.path.basename(a.model),
                         variant="_onestep" if a.onestep_latent else "")
        if a.onestep_latent and not a.spawn_server:
            print("[main] NOTE: --onestep-latent only labels the cache here; make sure YOUR server was started "
                  "with `--kwarg onestep_latent=True`, otherwise normal-quality chunks get cached as one-step.")
        if a.spawn_server:
            port = a.url.rsplit(":", 1)[-1].strip("/")
            cmd = [sys.executable, "-m", "terrain_diffusion", "api", a.model,
                   "--seed", str(a.seed), "--port", port]
            if not a.compile:
                cmd.append("--no-compile")
            if a.onestep_latent:
                cmd += ["--kwarg", "onestep_latent=True"]
            cmd += shlex.split(a.server_args)
            print("[main] starting server:", " ".join(cmd))
            proc = subprocess.Popen(cmd, cwd=os.path.expanduser(a.repo))
            atexit.register(lambda: proc.poll() is None and proc.terminate())
        print(f"[main] waiting for terrain server at {a.url} ...")
        if not src.wait_ready(timeout=600, alive=(lambda: proc.poll() is None) if proc else (lambda: True)):
            sys.exit(f"Cannot reach terrain server at {a.url}. Start it with:\n"
                     f"  python -m terrain_diffusion api {a.model} --seed {a.seed}\n"
                     f"or run this script with --spawn-server --repo <path to repo>, or use --demo.")
        print("[main] server up; first request loads the model (can take a while)...")

    r_pre = a.radius + a.prefetch
    mgr = ChunkManager(src, chunk=a.chunk, r_active=a.radius, r_prefetch=r_pre,
                       r_evict=r_pre + a.evict, cache_dir=a.cache_dir, workers=a.workers)
    mgr.pregen_radius = a.pregen if a.pregen is not None else a.radius

    w, h = a.size.split("x")
    loadPrcFileData("", f"win-size {w} {h}\nwindow-title Terrain Diffusion Explorer\n"
                        "framebuffer-multisample 1\nmultisamples 4\nsync-video 1\n")
    if a.selftest:
        loadPrcFileData("", "window-type offscreen\naudio-library-name null\n")
    from app import Explorer
    si, sj = (float(x) for x in a.start.split(","))
    app = Explorer(mgr, mpp=a.mpp, vexag=a.vexag, speed=a.speed, start=(si, sj))
    if a.selftest:
        import time as _t
        app.pitch = -22
        app.pos[0] += 5 * a.chunk            # teleport 5 chunks east: ring must re-stream
        app.pos[2] += 120
        for _ in range(a.selftest):
            app.taskMgr.step()
            _t.sleep(0.02)
        c = app._chunk_of_pos()
        from chunks import keys_within
        ring = keys_within(c, a.radius)
        missing = [k for k in ring if k not in app.meshes]
        print(f"[main] after teleport: center {c}, drawn {len(app.meshes)}, ring {len(ring)}, missing {len(missing)}")
        app.screenshot("selftest.png", defaultFilename=False)
        app.taskMgr.step()
        print("[main] selftest done", mgr.stats())
        mgr.close()
        return
    try:
        app.run()
    finally:
        mgr.close()


if __name__ == "__main__":
    main()