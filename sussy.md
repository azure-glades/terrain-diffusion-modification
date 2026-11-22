All of these run from the project root with `.venv/bin/python`. The new three-stage setup can't run yet, because `runs/struct`, `runs/d1` and `runs/d2` don't exist until you train them.

## Teacher (the real Terrain Diffusion model)

The explorer starts the model server itself:
```bash
.venv/bin/python main.py --spawn-server --repo terrain-diffusion --seed 42 --server-args "--dtype bf16"
```

Or start the server yourself in one terminal, then the explorer in another. The seeds must match:
```bash
.venv/bin/python -m terrain_diffusion api xandergos/terrain-diffusion-30m --seed 42 --dtype bf16
```
```bash
.venv/bin/python main.py --seed 42
```

Adding `--onestep-latent` gives a faster, lower-quality teacher.

## Students

**Classic four students on the procedural level 0** (deterministic L1/L2):
```bash
.venv/bin/python main.py --student runs/l1/best.pt,runs/l2/best.pt,runs/l3/best.pt,runs/l4/best.pt --radius 4 --pregen 3
```

**Same, with the noise-driven (GAN) L1/L2**, which is the setup measured earlier:
```bash
.venv/bin/python main.py --student runs/l1_gan/last.pt,runs/l2_gan/last.pt,runs/l3/best.pt,runs/l4/best.pt --radius 4 --pregen 3
```

**Hybrid: the teacher supplies the coarse structure and the students add the detail.** This needs the teacher server, so it isn't fast, but it shows how good the students look when given real structure:
```bash
.venv/bin/python main.py --student runs/l3/best.pt,runs/l4/best.pt --student-l0 teacher --student-start 2 --spawn-server --repo terrain-diffusion
```

**Shared model** (one network for all four levels):
```bash
.venv/bin/python main.py --student runs/shared/best.pt
```

**Older L3/L4 versions:** swap in `runs/l3_v1/best.pt` and `runs/l4_v1/best.pt`.

**After you run `train_v2.sh`**, the new three-stage generator is:
```bash
.venv/bin/python main.py --student runs/d1/last.pt,runs/d2/last.pt --structure runs/struct/best.pt --repo terrain-diffusion
```

## Other options

- **Which checkpoint:** any `best.pt`, `last.pt` or `snap_*.pt` can go in `--student`, and checkpoints from different runs can be mixed. Every level from the start level to 30 m needs one.
- **Where the cascade starts:** `--student-start 1|2|3` together with `--student-l0 teacher` gives more teacher and fewer students.
- **Procedural level 0:** `--student-l0-relief 0.5` flattens the ridged noise.
- **Zero-shot test:** `--student-reuse-net runs/l1_gan/last.pt` runs one network at every level.
- **Speed and quality:**
  - `--student-dtype bf16`.
  - `--student-block 512` and `--student-halo N`. A smaller halo is faster but can show seams.
  - For the new setup, `--structure-steps`, `--structure-block` and `--structure-halo`.
- **World and view:**
  - `--seed`, `--start row,col`, `--radius`, `--prefetch`, `--pregen` and `--workers`.
  - `--vexag` sets vertical exaggeration and `--speed` the fly speed.
  - `--size 1920x1080` sets the window size and `--chunk` the chunk size.
- **No model at all:** `--demo` uses a procedural stand-in.

Each source configuration gets its own folder under `cache/`, so switching between modes won't mix terrain. Delete that folder if you want a cold run for timing.

## Comparing without flying around

```bash
.venv/bin/python show_levels.py --data data/val2 --students runs/l1_gan/last.pt,runs/l2_gan/last.pt,runs/l3/best.pt,runs/l4/best.pt --sample 0,3 --out levels.png
```
```bash
.venv/bin/python cascade_eval.py --data data/val2 --students runs/l1_gan/last.pt,runs/l2_gan/last.pt,runs/l3/best.pt,runs/l4/best.pt --samples 8 --png cascade.png
```

The first shows the teacher, the cascade and each student on teacher input side by side for every level. The second gives the numbers. Both also take `--start-level 2` to begin from the teacher's level 2.