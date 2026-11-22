#!/usr/bin/env python3
"""Teacher-data pyramid generator for training a stateless coarse-to-fine terrain student.

Two jobs:
  generate : run the Terrain Diffusion teacher on random far-apart regions, cut non-overlapping
             patches, filter dull ones, and store them as resumable shards.
  PyramidData : reads the shards and builds, on the fly, a Laplacian-style pyramid per patch:
             levels (coarse->fine, 2x apart: 480 m ... 30 m for 5 levels) + residual bands
             r_0 = x_0,  r_l = x_l - up2(x_{l-1}).  The student at level l learns to produce r_l from up2(x_{l-1}).

Usage (run from your project dir; --repo points at the terrain-diffusion clone):
  python make_pyramid_data.py generate --repo terrain-diffusion/ --out data/pyr --blocks 50
  python make_pyramid_data.py inspect  --out data/pyr --png data/pyr_preview.png
  python make_pyramid_data.py generate --demo --out /tmp/pyr_demo --blocks 3 --block 512 --patch 128   # no GPU/model
  python make_pyramid_data.py selftest                                                                  # sanity checks
  python make_pyramid_data.py add-coarse --repo terrain-diffusion/ --out data/train3   # teacher coarse maps for existing shards

Coarse conditioning (for the structure model, see structure_model.py / train_structure.py): next to every shard
block_k.npz a sidecar block_k.coarse.npz holds, per patch, the teacher's own coarse-stage map (6 channels at
256 finest px = 7.68 km per pixel) over the patch footprint plus a margin of COARSE_MARGIN coarse pixels. `generate`
writes it for new blocks; `add-coarse` re-derives it for shards made earlier (cheap: only the teacher's coarse stage
runs, from the stored origins and seed).

Training side:
  from make_pyramid_data import PyramidData
  ds = PyramidData("data/pyr", levels=5, crop=256)      # any map-style dataset works with torch DataLoader
  item = ds[0]   # {'levels': [(1,h,w)...coarse->fine], 'residuals': [...], 'origin': (2,), ['climate': (4,h,w)]}

Notes
  * Defaults use the full-quality teacher (no one-step latent): its outputs ARE the training targets.
  * Each block is ONE large teacher request (cheaper per megapixel than many small cold ones); patches are cut from it.
  * Everything is deterministic: block k always maps to the same location, so runs resume and can be regenerated
    from the manifest (origins are stored, so you can later re-query the teacher for overlapping windows to test seams).
"""
import argparse
import collections
import json
import os
import sys
import time

import numpy as np

# =============================================================== pyramid math (numpy only)


def down2(x):
    """2x2 area average. x: (..., h, w) with even h, w."""
    h, w = x.shape[-2:]
    assert h % 2 == 0 and w % 2 == 0, f"size {h}x{w} not divisible by 2"
    return x.reshape(*x.shape[:-2], h // 2, 2, w // 2, 2).mean(axis=(-3, -1))


def _up_axis(a, axis):
    n = a.shape[axis]
    src = (np.arange(2 * n) + 0.5) / 2 - 0.5                 # bilinear, align_corners=False (matches torch)
    i0 = np.floor(src).astype(np.int64)
    f = (src - i0).astype(a.dtype)
    i1 = np.clip(i0 + 1, 0, n - 1)
    i0 = np.clip(i0, 0, n - 1)
    shape = [1] * a.ndim
    shape[axis] = -1
    f = f.reshape(shape)
    return np.take(a, i0, axis=axis) * (1 - f) + np.take(a, i1, axis=axis) * f


def up2(x):
    return _up_axis(_up_axis(x, -2), -1)


def build_pyramid(x, levels):
    """Returns [x_0 (coarsest) ... x_{levels-1} (= x, finest)]; each level is a 2x2 average of the next."""
    if x.shape[-1] % (2 ** (levels - 1)) or x.shape[-2] % (2 ** (levels - 1)):
        raise ValueError(f"patch {x.shape[-2:]} must be divisible by {2 ** (levels - 1)} for {levels} levels")
    out = [x]
    for _ in range(levels - 1):
        out.append(down2(out[-1]))
    return out[::-1]


def residual_bands(levels):
    """Laplacian-style bands: r_0 = x_0, r_l = x_l - up2(x_{l-1}). Perfectly invertible."""
    return [levels[0]] + [levels[i] - up2(levels[i - 1]) for i in range(1, len(levels))]


def reconstruct(res):
    x = res[0]
    for r in res[1:]:
        x = up2(x) + r
    return x


def encode_elev(x, space="sqrt", norm=50.0):
    """metres -> network space. 'sqrt' = signed square root (tames the huge dynamic range), then / norm."""
    if space == "sqrt":
        return np.sign(x) * np.sqrt(np.abs(x)) / norm
    return x / norm


def decode_elev(y, space="sqrt", norm=50.0):
    if space == "sqrt":
        v = y * norm
        return np.sign(v) * v * v
    return y * norm


# =============================================================== dataset


class PyramidData:
    """Map-style dataset over the shards written by `generate`."""

    def __init__(self, root, levels=5, crop=None, space="sqrt", norm=50.0, augment=True,
                 with_climate=False, cache_shards=4, preload=False):
        self.root = root
        self.man = load_manifest(root)
        if self.man is None:
            raise FileNotFoundError(f"no manifest.json in {root}")
        self.levels, self.crop, self.space, self.norm = levels, crop, space, norm
        self.augment = augment
        self.with_climate = with_climate and self.man["with_climate"]
        if crop is not None and crop % (2 ** (levels - 1)):
            raise ValueError(f"crop must be divisible by {2 ** (levels - 1)}")
        self.index = [(si, k) for si, s in enumerate(self.man["shards"]) for k in range(s["n"])]
        self._cache = collections.OrderedDict()
        self._cache_n = cache_shards
        # preload=True keeps every patch in RAM (int16: ~0.5 MB per 512px patch). Use it for training:
        # random access into compressed shards would otherwise decompress a whole shard per sample.
        self._pre = None
        if preload:
            el, orig, cl = [], [], []
            for s in self.man["shards"]:
                if s["n"] == 0:
                    continue
                z = np.load(os.path.join(root, s["file"]))
                el.append(z["elev"])
                orig.append(z["origins"])
                if self.with_climate:
                    cl.append(z["climate"])
            self._pre = dict(elev=np.concatenate(el) if el else np.zeros((0, 1, 1), np.int16),
                             origins=np.concatenate(orig) if orig else np.zeros((0, 2), np.int64),
                             climate=np.concatenate(cl) if cl else None)

    def __len__(self):
        return len(self.index)

    def _shard(self, si):
        if si in self._cache:
            self._cache.move_to_end(si)
            return self._cache[si]
        z = np.load(os.path.join(self.root, self.man["shards"][si]["file"]))
        d = {k: z[k] for k in z.files}
        self._cache[si] = d
        if len(self._cache) > self._cache_n:
            self._cache.popitem(last=False)
        return d

    def __getitem__(self, i):
        if self._pre is not None:
            sh, k = self._pre, i
        else:
            si, k = self.index[i]
            sh = self._shard(si)
        x = sh["elev"][k].astype(np.float32)
        c = sh["climate"][k].astype(np.float32) if self.with_climate else None
        P = x.shape[-1]
        rng = np.random.default_rng()
        if self.crop is not None and self.crop < P:
            a, b = rng.integers(0, P - self.crop + 1, size=2)
            x = x[a:a + self.crop, b:b + self.crop]
            c = c[:, a:a + self.crop, b:b + self.crop] if c is not None else None
        if self.augment:                                      # 8 dihedral symmetries
            if rng.random() < 0.5:
                x = x[:, ::-1]
                c = c[:, :, ::-1] if c is not None else None
            r = int(rng.integers(0, 4))
            x = np.rot90(x, r)
            c = np.rot90(c, r, axes=(1, 2)) if c is not None else None
        x = np.ascontiguousarray(x)
        lv = [encode_elev(l, self.space, self.norm)[None] for l in build_pyramid(x, self.levels)]
        item = {"levels": [l.astype(np.float32) for l in lv],
                "residuals": [r.astype(np.float32) for r in residual_bands(lv)],
                "origin": np.asarray(sh["origins"][k], dtype=np.int64)}
        if c is not None:
            item["climate"] = np.ascontiguousarray(c)
        return item


COARSE_PX = 256          # finest (30 m) pixels per teacher coarse pixel
COARSE_MARGIN = 2        # coarse pixels of context stored around every patch
COARSE_CH = 6


def coarse_file(shard_file):
    return shard_file[:-len(".npz")] + ".coarse.npz"


class StructData:
    """Training data for the structure model, all in RAM: per patch the level-0 map (teacher output averaged to
    `down` x 30 m = 480 m, encoded) and the teacher's coarse map around it.

    item = {'x': (1, S, S) encoded level 0, 'coarse': (6, S/cpp + 2*margin, same) raw coarse channels}
    with S = patch / down and cpp = COARSE_PX / down level-0 px per coarse px. Patches smaller than `min_px`
    level-0 pixels are skipped (512 px patches are only 32 px at 480 m: too small to learn structure from)."""

    def __init__(self, root, space="sqrt", norm=50.0, augment=True, down=16, min_px=64):
        man = load_manifest(root)
        if man is None:
            raise FileNotFoundError(f"no manifest.json in {root}")
        if not man.get("coarse"):
            raise FileNotFoundError(f"{root} has no teacher coarse maps yet; run:  python make_pyramid_data.py "
                                    f"add-coarse --repo <terrain-diffusion clone> --out {root}")
        P = man["patch"]
        if P % COARSE_PX or COARSE_PX % down:
            raise ValueError(f"patch {P} must be a multiple of {COARSE_PX} and down ({down}) must divide {COARSE_PX}")
        self.margin, self.cpp, self.S = int(man["coarse"]["margin"]), COARSE_PX // down, P // down
        self.augment, self.man = augment, man
        xs, cs = [], []
        if self.S >= min_px:
            for s in man["shards"]:
                if s["n"] == 0:
                    continue
                e = np.load(os.path.join(root, s["file"]))["elev"].astype(np.float32)
                e = e.reshape(len(e), self.S, down, self.S, down).mean(axis=(2, 4))
                xs.append(encode_elev(e, space, norm).astype(np.float32))
                cs.append(np.load(os.path.join(root, coarse_file(s["file"])))["coarse"].astype(np.float32))
        n = self.S // self.cpp + 2 * self.margin
        self.x = np.concatenate(xs) if xs else np.zeros((0, self.S, self.S), np.float32)
        self.c = np.concatenate(cs) if cs else np.zeros((0, COARSE_CH, n, n), np.float32)
        assert len(self.x) == len(self.c) and self.c.shape[1:] == (COARSE_CH, n, n), (self.x.shape, self.c.shape)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        x, c = self.x[i], self.c[i]
        if self.augment:                                      # the same dihedral symmetry on both grids
            rng = np.random.default_rng()
            if rng.random() < 0.5:
                x, c = x[:, ::-1], c[:, :, ::-1]
            r = int(rng.integers(0, 4))
            x, c = np.rot90(x, r), np.rot90(c, r, axes=(1, 2))
        return {"x": np.ascontiguousarray(x)[None], "coarse": np.ascontiguousarray(c)}


# =============================================================== manifest


def manifest_path(root):
    return os.path.join(root, "manifest.json")


def load_manifest(root):
    p = manifest_path(root)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save_manifest(root, man):
    tmp = manifest_path(root) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1)
    os.replace(tmp, manifest_path(root))


# =============================================================== teachers


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x)


class DemoTeacher:
    """Procedural stand-in so the data path and training code can be tested without GPU/model."""

    def __init__(self, seed):
        here = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, here)
        from sources import DemoSource
        self.src = DemoSource(seed)

    def get(self, i1, j1, i2, j2, climate=False):
        e = self.src.get(i1, j1, i2, j2).astype(np.float32)
        c = None
        if climate:
            c = np.stack([15 - 6.5 * np.maximum(e, 0) / 1000, np.full_like(e, 5.0),
                          np.full_like(e, 800.0), np.full_like(e, 30.0)])
        return e, c

    def coarse(self, ci1, cj1, ci2, cj2):
        """Stand-in for the teacher's coarse stage: block statistics of the demo terrain (16 x 16 samples per pixel)."""
        k, st = 16, COARSE_PX // 16
        ii = np.arange(ci1 * COARSE_PX, ci2 * COARSE_PX, st) + st / 2
        jj = np.arange(cj1 * COARSE_PX, cj2 * COARSE_PX, st) + st / 2
        z = self.src.sample(ii, jj).reshape(ci2 - ci1, k, cj2 - cj1, k).transpose(0, 2, 1, 3).reshape(ci2 - ci1, cj2 - cj1, -1)
        ssq = lambda v: np.sign(v) * np.sqrt(np.abs(v))        # noqa: E731
        mean = z.mean(-1)
        out = np.stack([ssq(mean), ssq(np.percentile(z, 5, axis=-1)), 15 - 6.5 * np.maximum(mean, 0) / 1000,
                        np.full_like(mean, 600.0), np.full_like(mean, 800.0), np.full_like(mean, 60.0)])
        return out.astype(np.float32)


class PipelineTeacher:
    def __init__(self, a, coarse_only=False):
        """coarse_only: keep just the coarse stage (drops the latent and decoder networks; get() must not be used)
        and restore the working directory afterwards -- this is how the explorer uses the teacher's coarse map."""
        cwd0 = os.getcwd()
        repo = os.path.abspath(os.path.expanduser(a.repo))
        if not os.path.isdir(os.path.join(repo, "terrain_diffusion")):
            sys.exit(f"--repo {repo} does not look like the terrain-diffusion clone")
        os.chdir(repo)                                         # repo loads data/ and checkpoints/ relative to its root
        sys.path.insert(0, repo)
        import inspect as _inspect
        import torch
        from terrain_diffusion.inference.world_pipeline import WorldPipeline
        batch = [int(x) for x in a.batch.split(",")] if "," in a.batch else int(a.batch)
        kw = dict(seed=a.seed, dtype=None if a.dtype == "fp32" else a.dtype, torch_compile=a.compile,
                  latents_batch_size=batch, caching_strategy="direct",
                  cache_limit=int(a.cache_gb * 1024 ** 3), log_mode="info")
        sig = _inspect.signature(WorldPipeline.from_pretrained)
        if not any(p.kind == _inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            kw = {k: v for k, v in kw.items() if k in sig.parameters}
        t0 = time.time()
        self.pipe = WorldPipeline.from_pretrained(a.model, **kw)
        if coarse_only:
            self.pipe.base_model = None
            self.pipe.decoder_model = None
        dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.pipe = self.pipe.to(dev) or self.pipe
        except Exception as e:
            print(f"[warn] pipe.to({dev}) failed: {e!r}")
        for call in (lambda: self.pipe.bind(), lambda: self.pipe.bind(hdf5_file="TEMP")):
            try:
                call()
            except Exception:
                continue
            if getattr(self.pipe, "residual", None) is not None:
                break
        else:
            if getattr(self.pipe, "residual", None) is None:
                sys.exit("could not bind the pipeline (see profile_terrain.py ensure_bound for details)")
        if coarse_only:
            os.chdir(cwd0)
        print(f"[teacher] {'coarse stage ' if coarse_only else ''}ready on {dev} in {time.time() - t0:.1f}s")

    def coarse(self, ci1, cj1, ci2, cj2):
        """The teacher's coarse-stage map for coarse pixels [ci1:ci2) x [cj1:cj2) (one pixel = COARSE_PX finest px):
        (6, h, w) = [elev, p5 elev] in signed-sqrt metres + 4 climate channels. Deterministic in the seed."""
        w = _np(self.pipe.coarse[:, ci1:ci2, cj1:cj2])
        return (w[:-1] / w[-1:]).astype(np.float32)

    def get(self, i1, j1, i2, j2, climate=False):
        try:
            out = self.pipe.get(i1, j1, i2, j2, with_climate=climate)
        except TypeError:
            out = self.pipe.get(i1, j1, i2, j2)
        e = _np(out["elev"]).astype(np.float32)
        c = _np(out["climate"]).astype(np.float32) if climate and "climate" in out else None
        return e, c


# =============================================================== generate


def write_coarse(teacher, out, shard_file, origins):
    """Sidecar with the teacher's coarse map around every patch of a shard (origins: (n, 2) finest px)."""
    M, hc = COARSE_MARGIN, None
    maps = []
    for i0, j0 in np.asarray(origins).reshape(-1, 2):
        ci, cj = int(i0) // COARSE_PX, int(j0) // COARSE_PX
        hc = teacher.patch // COARSE_PX
        maps.append(teacher.coarse(ci - M, cj - M, ci + hc + M, cj + hc + M))
    n = (hc or 0) + 2 * M
    arr = np.stack(maps) if maps else np.zeros((0, COARSE_CH, n, n), np.float32)
    dst = os.path.join(out, coarse_file(shard_file))
    tmp = dst[:-len(".npz")] + ".tmp.npz"
    np.savez_compressed(tmp, coarse=arr.astype(np.float32))
    os.replace(tmp, dst)


def cmd_add_coarse(a):
    out = os.path.abspath(os.path.expanduser(a.out))
    man = load_manifest(out)
    if man is None:
        sys.exit(f"no manifest.json in {out}")
    if man["patch"] % COARSE_PX:
        sys.exit(f"patch {man['patch']} is not a multiple of {COARSE_PX}: no coarse grid alignment")
    redo = getattr(a, "redo", False)
    todo = [s for s in man["shards"] if s["n"] > 0
            and (redo or not os.path.exists(os.path.join(out, coarse_file(s["file"]))))]
    print(f"[coarse] {len(todo)} of {len(man['shards'])} shards need a coarse sidecar "
          f"(teacher seed {man['seed']}, model {man['model']}, dtype {man['dtype']})")
    if todo:
        ns = argparse.Namespace(**{**vars(a), "seed": man["seed"], "model": man["model"], "dtype": man["dtype"]})
        teacher = DemoTeacher(man["seed"]) if man.get("demo") else PipelineTeacher(ns, coarse_only=True)
        teacher.patch = man["patch"]
        t0 = time.time()
        for k, s in enumerate(todo):
            write_coarse(teacher, out, s["file"], np.load(os.path.join(out, s["file"]))["origins"])
            if (k + 1) % 20 == 0 or k + 1 == len(todo):
                print(f"[coarse] {k + 1}/{len(todo)} shards  ({time.time() - t0:.0f}s)", flush=True)
    # The coarse maps are only useful if they are the ones this dataset's terrain was generated FROM. The teacher's
    # coarse stage is a 20-step sampler and does not reproduce across devices / precisions (CPU fp32 and GPU bf16 give
    # different worlds), so check: the coarse elevation must track the block means of the stored terrain.
    M, ce, te = COARSE_MARGIN, [], []
    full = [s for s in man["shards"] if s["n"] > 0]
    for s in full[::max(1, len(full) // 40)]:
        c = np.load(os.path.join(out, coarse_file(s["file"])))["coarse"][:, 0, M:-M, M:-M]
        e = np.load(os.path.join(out, s["file"]))["elev"].astype(np.float32)
        h = c.shape[-1]
        ce.append((np.sign(c) * c * c).ravel())
        te.append(e.reshape(len(e), h, COARSE_PX, h, COARSE_PX).mean(axis=(2, 4)).ravel())
    corr = float(np.corrcoef(np.concatenate(ce), np.concatenate(te))[0, 1]) if ce else 1.0
    print(f"[coarse] check: correlation between coarse elevation and the stored terrain's 7.68 km means = {corr:.3f}")
    if corr < 0.7 and not getattr(a, "force", False):
        man.pop("coarse", None)
        save_manifest(out, man)
        sys.exit(f"[coarse] MISMATCH: this teacher run does not reproduce the world {out} was generated from.\n"
                 f"  Run add-coarse on the same machine / GPU and settings that generated the data (dtype "
                 f"{man['dtype']}), then repeat with --redo to overwrite these sidecars. (--force accepts them anyway.)")
    man["coarse"] = dict(margin=COARSE_MARGIN, px=COARSE_PX, channels=COARSE_CH, check_corr=round(corr, 3))
    save_manifest(out, man)
    print(f"[done] coarse maps complete in {out}")


def cmd_generate(a):
    out = os.path.abspath(os.path.expanduser(a.out))                 # resolve BEFORE the teacher chdirs into the repo
    os.makedirs(out, exist_ok=True)
    B, P = a.block, a.patch
    if B % P:
        sys.exit("--block must be a multiple of --patch")
    if P % (2 ** (a.levels - 1)):
        sys.exit(f"--patch must be divisible by {2 ** (a.levels - 1)} for {a.levels} levels")
    cfg = dict(patch=P, block=B, seed=a.seed, data_seed=a.data_seed, levels=a.levels, model=a.model,
               dtype=a.dtype, mpp=30.0, with_climate=bool(a.with_climate), demo=bool(a.demo),
               min_std=a.min_std, max_ocean=a.max_ocean, span=a.span)
    man = load_manifest(out)
    if man is None:
        man = dict(version=1, **cfg, shards=[], skipped_patches=0, kept_patches=0)
    else:
        for k in ("patch", "block", "seed", "data_seed", "model", "with_climate", "demo"):
            if man.get(k) != cfg[k]:
                sys.exit(f"existing dataset in {out} was made with {k}={man.get(k)!r}, now {cfg[k]!r}; use a new --out")
        man.update(min_std=a.min_std, max_ocean=a.max_ocean)
    done = {s["block"] for s in man["shards"]}

    teacher = DemoTeacher(a.seed) if a.demo else PipelineTeacher(a)
    teacher.patch = P
    with_coarse = P % COARSE_PX == 0                                 # coarse sidecars need coarse-grid aligned patches
    if with_coarse and not man["shards"]:
        man["coarse"] = dict(margin=COARSE_MARGIN, px=COARSE_PX, channels=COARSE_CH)
    G = max(1, a.span // B)
    npb = (B // P) ** 2
    t_start, blocks_run = time.time(), 0
    for k in range(a.blocks):
        if k in done:
            continue
        rng = np.random.default_rng([a.data_seed, k])
        gi, gj = (int(v) for v in rng.integers(-G, G + 1, size=2))
        i0, j0 = gi * B, gj * B
        if getattr(a, "probe", 0):
            c0 = a.probe
            pi, pj = i0 + B // 2 - c0 // 2, j0 + B // 2 - c0 // 2
            pe, _ = teacher.get(pi, pj, pi + c0, pj + c0, climate=False)
            if (np.asarray(pe) <= 0).mean() > min(a.max_ocean + a.probe_margin, 0.98):
                np.savez_compressed(os.path.join(out, f"block_{k:06d}.npz"), elev=np.zeros((0, P, P), np.int16),
                                    origins=np.zeros((0, 2), np.int64),
                                    **({"climate": np.zeros((0, 4, P, P), np.float16)} if a.with_climate else {}))
                man["shards"].append(dict(file=f"block_{k:06d}.npz", n=0, block=k, origin=[i0, j0], probed=True))
                man["skipped_patches"] += npb
                save_manifest(out, man)
                print(f"[gen] block {k + 1}/{a.blocks} at ({i0},{j0}): probe says mostly ocean, skipped", flush=True)
                continue
        t0 = time.time()
        elev, clim = teacher.get(i0, j0, i0 + B, j0 + B, climate=bool(a.with_climate))
        gen_s = time.time() - t0
        e16 = np.clip(np.rint(elev), -32768, 32767).astype(np.int16)
        keep, origins = [], []
        for r in range(B // P):
            for c in range(B // P):
                p = e16[r * P:(r + 1) * P, c * P:(c + 1) * P]
                if (p <= 0).mean() > a.max_ocean or float(p.std()) < a.min_std:
                    man["skipped_patches"] += 1
                    continue
                keep.append((r, c))
                origins.append((i0 + r * P, j0 + c * P))
        arrs = dict(elev=np.stack([e16[r * P:(r + 1) * P, c * P:(c + 1) * P] for r, c in keep])
                    if keep else np.zeros((0, P, P), np.int16),
                    origins=np.asarray(origins, dtype=np.int64).reshape(-1, 2))
        if a.with_climate and clim is not None:
            arrs["climate"] = (np.stack([clim[:, r * P:(r + 1) * P, c * P:(c + 1) * P] for r, c in keep]).astype(np.float16)
                               if keep else np.zeros((0, 4, P, P), np.float16))
        fname = f"block_{k:06d}.npz"
        tmp = os.path.join(out, f"block_{k:06d}.tmp.npz")
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, os.path.join(out, fname))
        if with_coarse and keep:
            write_coarse(teacher, out, fname, arrs["origins"])
        man["shards"].append(dict(file=fname, n=len(keep), block=k, origin=[i0, j0]))
        man["kept_patches"] += len(keep)
        save_manifest(out, man)
        blocks_run += 1
        el = time.time() - t_start
        eta = el / blocks_run * (a.blocks - len(done) - blocks_run)
        print(f"[gen] block {k + 1}/{a.blocks} at ({i0},{j0}): teacher {gen_s:.1f}s "
              f"({B * B / 1e6 / max(gen_s, 1e-9):.2f} Mpx/s), kept {len(keep)}/{npb} patches, "
              f"total {man['kept_patches']}  ETA {eta / 60:.1f} min", flush=True)
    gb = man["kept_patches"] * P * P * 2 / 1e9
    print(f"[done] {man['kept_patches']} patches ({man['skipped_patches']} filtered) in {out}  "
          f"~{gb:.2f} GB uncompressed int16")


# =============================================================== inspect / selftest


def hillshade(z, mpp, vexag=2.0):
    gy, gx = np.gradient(z * vexag / mpp)
    nz = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    nx, ny = -gx * nz, -gy * nz
    L = np.array([-0.5, 0.5, 0.7])
    L = L / np.linalg.norm(L)
    return np.clip(nx * L[0] + ny * L[1] + nz * L[2], 0, 1)


def cmd_inspect(a):
    root = os.path.abspath(os.path.expanduser(a.out))
    man = load_manifest(root)
    if man is None:
        sys.exit(f"no manifest.json in {root}")
    levels = a.levels or man["levels"]
    ds = PyramidData(root, levels=levels, augment=False, space="metres", norm=1.0)
    print(f"dataset: {len(ds)} patches of {man['patch']}px  (blocks: {len(man['shards'])}, filtered out: {man['skipped_patches']})")
    if not len(ds):
        return
    rng = np.random.default_rng(0)
    pick = rng.choice(len(ds), size=min(a.samples, len(ds)), replace=False)
    worst, stds, ocean, rows = 0.0, [], [], []
    for i in pick:
        it = ds[int(i)]
        fine = it["levels"][-1][0]
        rec = reconstruct([r[0] for r in it["residuals"]])
        worst = max(worst, float(np.abs(rec - fine).max()))
        stds.append(float(fine.std()))
        ocean.append(float((fine <= 0).mean()))
        rows.append([it["levels"][l][0] for l in range(levels)])
    print(f"sampled {len(pick)}: elevation std median {np.median(stds):.0f} m, mean ocean fraction {np.mean(ocean):.2f}")
    print(f"pyramid reconstruction max error: {worst:.2e} m  ({'OK' if worst < 1e-2 else 'PROBLEM'})")
    res_std = [np.mean([np.std(residual_bands([ds[int(i)]['levels'][l][0] for l in range(levels)])[b]) for i in pick[:8]])
               for b in range(levels)]
    print("residual band std (m), coarse->fine: " + "  ".join(f"{s:.1f}" for s in res_std))
    if a.png:
        try:
            from PIL import Image
        except ImportError:
            print("[note] Pillow not installed; skipping --png (pip install pillow)")
            return
        S = 192
        tiles = []
        for lv in rows:
            row = []
            for l, x in enumerate(lv):
                mpp = man["mpp"] * 2 ** (levels - 1 - l)
                h = (hillshade(x, mpp) * 255).astype(np.uint8)
                row.append(np.asarray(Image.fromarray(h).resize((S, S), Image.NEAREST)))
            tiles.append(np.concatenate(row, axis=1))
        Image.fromarray(np.concatenate(tiles, axis=0)).save(a.png)
        print(f"[png] wrote {a.png}  (columns: coarse -> fine, each upscaled to {S}px)")


def cmd_selftest(a):
    import shutil
    import tempfile
    x = np.random.default_rng(1).normal(size=(2, 64, 64)).astype(np.float32)
    assert np.allclose(down2(x)[0, 0, 0], x[0, :2, :2].mean())
    c = np.full((8, 8), 3.0, np.float32)
    assert np.allclose(up2(c), 3.0)                                # constants are preserved
    lv = build_pyramid(x[0], 5)
    assert [l.shape[-1] for l in lv] == [4, 8, 16, 32, 64]
    assert np.abs(reconstruct(residual_bands(lv)) - lv[-1]).max() < 1e-5
    y = np.array([-900.0, 0.0, 4.0, 8000.0])
    assert np.allclose(decode_elev(encode_elev(y)), y, atol=1e-3)
    d = tempfile.mkdtemp()
    try:
        ns = argparse.Namespace(**{**vars(a), "demo": True, "out": d, "blocks": 3, "block": 256, "patch": 64,
                                   "levels": 4, "with_climate": True, "min_std": 0.0, "max_ocean": 1.0, "seed": 0,
                                   "data_seed": 0, "span": 100000})
        cmd_generate(ns)
        cmd_generate(ns)                                           # resume: must skip finished blocks
        ds = PyramidData(d, levels=4, crop=32, with_climate=True)
        assert len(ds) == 3 * 16, len(ds)
        it = ds[5]
        assert [l.shape for l in it["levels"]] == [(1, 4, 4), (1, 8, 8), (1, 16, 16), (1, 32, 32)]
        assert it["climate"].shape == (4, 32, 32)
        pre = PyramidData(d, levels=4, crop=32, with_climate=True, preload=True)
        assert len(pre) == len(ds) and pre[7]["levels"][-1].shape == (1, 32, 32)
        assert np.abs(reconstruct([r[0] for r in it["residuals"]]) - it["levels"][-1][0]).max() < 1e-4
        # determinism: block k always maps to the same location / data
        m1 = load_manifest(d)["shards"][0]["origin"]
        d2 = tempfile.mkdtemp()
        cmd_generate(argparse.Namespace(**{**vars(ns), "out": d2}))
        assert load_manifest(d2)["shards"][0]["origin"] == m1
        shutil.rmtree(d2)
        cmd_inspect(argparse.Namespace(out=d, levels=4, samples=4, png=None))
        # coarse sidecars: written by generate for coarse-aligned patches, re-derived identically by add-coarse
        d3 = tempfile.mkdtemp()
        ns3 = argparse.Namespace(**{**vars(ns), "out": d3, "blocks": 2, "block": 512, "patch": 256, "levels": 5,
                                    "with_climate": False})
        cmd_generate(ns3)
        sd = StructData(d3, min_px=16)
        assert len(sd) == 8 and sd.cpp == 16, (len(sd), sd.cpp)
        it = sd[3]
        assert it["x"].shape == (1, 16, 16) and it["coarse"].shape == (COARSE_CH, 5, 5), (it["x"].shape, it["coarse"].shape)
        f0 = load_manifest(d3)["shards"][0]["file"]
        before = np.load(os.path.join(d3, coarse_file(f0)))["coarse"]
        os.remove(os.path.join(d3, coarse_file(f0)))
        cmd_add_coarse(argparse.Namespace(out=d3))
        assert np.array_equal(before, np.load(os.path.join(d3, coarse_file(f0)))["coarse"])
        shutil.rmtree(d3)
    finally:
        shutil.rmtree(d)
    print("SELFTEST PASSED")


# =============================================================== cli


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="run the teacher and write shards")
    g.add_argument("--out", required=True)
    g.add_argument("--repo", default=".", help="terrain-diffusion clone (not needed with --demo)")
    g.add_argument("--model", default="xandergos/terrain-diffusion-30m")
    g.add_argument("--demo", action="store_true", help="procedural stand-in teacher (no GPU/model)")
    g.add_argument("--blocks", type=int, default=20, help="number of teacher requests (resumable)")
    g.add_argument("--block", type=int, default=2048, help="px per teacher request (square); bigger = cheaper per Mpx")
    g.add_argument("--patch", type=int, default=512, help="px per stored training patch")
    g.add_argument("--levels", type=int, default=5, help="pyramid levels (2x apart): 5 -> 480,240,120,60,30 m")
    g.add_argument("--seed", type=int, default=0, help="world seed of the teacher")
    g.add_argument("--data-seed", type=int, default=0, help="seed for choosing block locations")
    g.add_argument("--span", type=int, default=200_000, help="blocks are placed within +-span pixels of the origin")
    g.add_argument("--min-std", type=float, default=3.0, help="drop patches flatter than this (metres std)")
    g.add_argument("--max-ocean", type=float, default=0.6, help="drop patches with more ocean than this fraction")
    g.add_argument("--probe", type=int, default=0,
                   help="before generating a block, request a small PxP window at its centre and skip the block if it is "
                        "mostly ocean (about 15x cheaper than a full block). 0 = off; try 256")
    g.add_argument("--probe-margin", type=float, default=0.15,
                   help="the probe skips when its ocean fraction exceeds --max-ocean + this margin (the probe is a noisy sample)")
    g.add_argument("--with-climate", action="store_true", help="also store temperature/precip channels (float16)")
    g.add_argument("--dtype", default="bf16", choices=["fp32", "bf16", "fp16"])
    g.add_argument("--batch", default="1,4,8,16")
    g.add_argument("--compile", action="store_true")
    g.add_argument("--cache-gb", type=float, default=1.0, help="teacher tile cache (blocks are far apart; keep it small)")
    g.add_argument("--device", default=None)
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("add-coarse", help="write the teacher's coarse maps for the shards of an existing dataset")
    c.add_argument("--out", required=True, help="dataset directory (its manifest supplies seed, model and dtype)")
    c.add_argument("--repo", default=".", help="terrain-diffusion clone")
    c.add_argument("--batch", default="1,4,8,16")
    c.add_argument("--compile", action="store_true")
    c.add_argument("--cache-gb", type=float, default=1.0)
    c.add_argument("--device", default=None)
    c.add_argument("--redo", action="store_true", help="overwrite existing sidecars")
    c.add_argument("--force", action="store_true", help="keep the coarse maps even if the consistency check fails")
    c.set_defaults(fn=cmd_add_coarse)

    i = sub.add_parser("inspect", help="stats, pyramid self-check, optional hillshade montage")
    i.add_argument("--out", required=True)
    i.add_argument("--levels", type=int, default=None)
    i.add_argument("--samples", type=int, default=8)
    i.add_argument("--png", default=None)
    i.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("selftest", help="numpy-only sanity checks using the demo teacher")
    s.set_defaults(fn=cmd_selftest, out=None, repo=".", model="x", dtype="fp32", batch="1", compile=False,
                   cache_gb=1.0, device=None, levels=4)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()