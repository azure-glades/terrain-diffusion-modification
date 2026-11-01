# Terrain Diffusion Explorer

clone https://github.com/xandergos/terrain-diffusion into this repository



Fly over an endless world generated on the fly by `xandergos/terrain-diffusion-30m`.
This app is a client of the repo's own API server (`python -m terrain_diffusion api ...`), so it
reuses the repo's model code, caching and batching. The server is deterministic and seamless
(InfiniteDiffusion), so neighbouring chunks line up with no stitching code here.

## Run

```bash
pip install -r requirements.txt

# 0) try it without the model (procedural stand-in, runs anywhere)
python main.py --demo

# 1) real model, app starts the server for you (needs your clone of the repo, deps installed there)
python main.py --spawn-server --repo ./terrain-diffusion --seed 42

# 2) or start the server yourself, then the app (seeds must match: it keys the disk cache)
python -m terrain_diffusion api xandergos/terrain-diffusion-30m --seed 42 --no-compile
python main.py --seed 42
```
To try out the student models:
```bash
python main.py --student runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt --radius 4 --pregen 3
```

First launch downloads the model and generates the start area before the window shows terrain.
Useful flags: `--radius 6` (drawn chunks), `--prefetch 2`, `--chunk 128`, `--workers 2`,
`--vexag 1.5`, `--start row,col`, `--server-args "--dtype bf16"`, `--compile` (torch.compile; off by default
because it can fail on Windows).

## Controls

WASD move, Space/E up, Q/C down, Shift fast, mouse wheel = speed, `-`/`=` vertical exaggeration,
Tab release/capture mouse, F wireframe, P screenshot, H hide HUD, Esc quit.

## How streaming works (chunks.py)

- A chunk = 128x128 heightmap px (~3.8 km at 30 m/px) plus a 1px apron (for seamless normals).
- Three circular rings around the player's chunk: **drawn** (`radius`), **prefetched** (`+prefetch`,
  heights generated but not meshed), **evicted** (`+evict` beyond that; heights dropped from RAM).
  Unload radii exceed load radii (hysteresis), so crossing a boundary back and forth doesn't thrash.
- Worker threads pull the nearest missing chunk, biased toward your direction of travel; the render
  thread only builds a few meshes per frame (6 ms budget), so it never stalls on generation.
- Every generated chunk is saved to `cache/<model_seed>/*.npy`; unloaded chunks reload from disk
  instead of the GPU.

## Things to check on first real run (I couldn't download the model in my environment)

- The HTTP payload layout (int16 elevation first, then float32 climate; headers X-Height/X-Width) is
  taken from the repo's documentation and mirrored in `tests.py`, not tested against the real server.
- If the world looks 3x too big/small, pass `--mpp` (metres per px) / server `--kwarg native_resolution=30`.
- Raise `--workers` if the GPU is idle while chunks queue; lower `--radius` if the HUD shows chunks
  queueing while you fly.

`python tests.py` runs headless tests (seams, mesh, streaming/eviction/disk reload, HTTP client).

# Misc
Best teacher config:
```claude
Teacher: your repo pipeline with bf16 and batch list 1,4,8,16, the best setup from your profiling. It times cold N×N requests at far-apart locations, so nothing is cached, and does a warm-up request first.
```