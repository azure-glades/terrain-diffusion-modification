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

This gives level-by-level comparision of terrain diffusion model and trained student model
```bash
python show_levels.py --data data/val2 \
  --students runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt \
  --sample 0,3 --out levels.png
```

Beyond the two, there are several other designs. Here are the ones that fit your goals (fast, stateless, random-access, “suitably realistic”) and what we’ve learned so far.

- Replace only the teacher’s decoder. In your profile the decoder took about half the time at 1024². Keep the teacher’s coarse and latent stages (which supply the realistic large-scale structure that the cascade lacks) and distill just the decoder into a small fast network. It’s a smaller step than replacing everything, and its quality ceiling is the teacher’s. The speedup would be modest (about 2×) unless the latent stage also gets faster.
- Fix the one-step latent with better distillation. You rejected onestep_latent for its quality drop, but that came from plain one-step sampling. Adversarial or distribution-matching distillation (the approaches behind the one-step image models) is designed to close that gap. It’s a research bet, and I haven’t seen it done on terrain.
- Diffusion or consistency model for the coarse levels only, with regressors for the fine ones. Levels 1–2 are tiny in pixels, so a real generative model there is cheap, and it removes the GAN stability risk. Levels 3–4 stay as they are, since they look good when given real structure.
- A latent-space generator. Train a high-compression autoencoder on terrain (32×–64× fewer pixels), generate in that latent space with a small model, and decode once. The decoder is convolutional, so it stays stateless with a halo. This is the VAE idea you started with, and it is the largest redesign.
- A convolutional GAN generator in the style of InfinityGAN. It’s one network that turns a noise grid plus coordinates into terrain, stateless by construction and with no cascade. It avoids exposure bias but depends entirely on adversarial training working.