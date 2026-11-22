"""Student cascade as an explorer terrain source.

Stateless and random-access by construction: the world is cut into fixed BLOCKs (default 512 px at the finest level).
A block is a pure function of (level-0 content, noise seed, block index): the cascade runs on a window padded by a halo
of >= the cascade's receptive-field margin, then the halo is cropped. Chunk requests are assembled from blocks, so
neighbouring chunks always agree and a block is generated once (LRU-cached) however many chunks need it.

Level-0 providers (anything with .window(level, F, a0, b0, n, m) -> float32 metres at level `level`):
  ProceduralL0 : fractal noise in the DemoSource style, point-sampled at 480 m/px (no model, no server)
  TeacherL0    : any TerrainSource (the real teacher via HttpSource, or DemoSource) block-averaged down to `level`
                 -> a hybrid: the teacher supplies the coarse structure, the students add the detail
  StructureL0  : the learned structure model (structure_model.py) on top of the teacher's coarse map
                 (TeacherCoarse: its synthetic map + coarse stage only, no latent/decoder network) -> 480 m level 0

The students may be the classic per-level ones or MERGED stages (one network for several levels); the chain is
assembled from each checkpoint's (src -> level).
"""
import argparse
import collections
import hashlib
import os
import threading
import time

import numpy as np

from sources import DemoSource, TerrainSource


class ProceduralL0:
    def __init__(self, seed=0, relief=1.0):
        self.demo = DemoSource(seed)
        self.relief = float(relief)                              # <1 flattens the world (the raw noise is very mountainous)
        self.tag = f"proc{seed}" + ("" if relief == 1.0 else f"_r{relief:g}")

    def window(self, level, F, a0, b0, n, m):
        f = 2 ** (F - level)                                   # finest px per level-`level` px
        ii = (np.arange(a0, a0 + n) + 0.5) * f - 0.5            # centre of each coarse pixel, in finest-px coordinates
        jj = (np.arange(b0, b0 + m) + 0.5) * f - 0.5
        return (self.relief * self.demo.sample(ii, jj)).astype(np.float32)


class TeacherL0:
    def __init__(self, src):
        self.src = src
        self.tag = f"teach_{src.tag}"

    def window(self, level, F, a0, b0, n, m):
        f = 2 ** (F - level)
        z = self.src.get(a0 * f, b0 * f, (a0 + n) * f, (b0 + m) * f).astype(np.float32)
        return z if f == 1 else z.reshape(n, f, m, f).mean(axis=(1, 3))


class TeacherCoarse:
    """The teacher's own coarse stage (procedural synthetic map + small coarse model) at 7.68 km/px."""

    def __init__(self, repo, seed=0, model="xandergos/terrain-diffusion-30m", dtype=None, device=None):
        from make_pyramid_data import PipelineTeacher
        if dtype is None:                                        # the training data's coarse maps were made in bf16
            import torch
            dtype = "bf16" if (device or ("cuda" if torch.cuda.is_available() else "cpu")).startswith("cuda") else "fp32"
        ns = argparse.Namespace(repo=repo, model=model, seed=int(seed), dtype=dtype, batch="1,4,8,16", compile=False,
                                cache_gb=0.25, device=device)
        self.t = PipelineTeacher(ns, coarse_only=True)
        self.tag = f"tcoarse{int(seed)}_{dtype}"

    def coarse(self, ci1, cj1, ci2, cj2):
        return self.t.coarse(ci1, cj1, ci2, cj2)


class DemoCoarse:
    """Procedural stand-in for TeacherCoarse (no model): block statistics of DemoSource."""

    def __init__(self, seed=0):
        from make_pyramid_data import DemoTeacher
        self.t = DemoTeacher(seed)
        self.tag = f"dcoarse{int(seed)}"

    def coarse(self, ci1, cj1, ci2, cj2):
        return self.t.coarse(ci1, cj1, ci2, cj2)


class StructureL0:
    """Level 0 (480 m/px) from the structure model. Stateless like the students: the world is cut into blocks of
    `block` coarse pixels; a block is generated on a window padded by the sampler's halo and cropped, so it depends
    only on the coarse map, the seed and its position. Blocks are LRU-cached (one covers block * 7.68 km)."""

    def __init__(self, ckpt, coarse_src, seed=0, steps=None, block=16, halo=None, device=None, dtype=None,
                 cache_blocks=64, log=print):
        import structure_model as stm
        import torch
        self.stm = stm
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, self.cfg = stm.load_structure(ckpt, self.dev)
        self.amp = torch.bfloat16 if dtype == "bf16" else None
        self.src, self.seed = coarse_src, int(seed)
        self.steps = int(steps or self.cfg.get("steps", 8))
        self.cpp, self.Bc = int(self.cfg["cpp"]), int(block)
        self.Hc = stm.halo_coarse(self.cfg, self.steps) if halo is None else int(halo)
        st = os.stat(ckpt)
        h = hashlib.sha1(f"{os.path.abspath(ckpt)}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:10]
        self.tag = f"struct_{h}_s{self.seed}_k{self.steps}_B{self.Bc}_H{self.Hc}_{coarse_src.tag}_{dtype or 'fp32'}"
        self._cache = collections.OrderedDict()
        self._max = cache_blocks
        self._lock = threading.Lock()
        self.block_ms = collections.deque(maxlen=20)
        n = (self.Bc + 2 * self.Hc) * self.cpp
        log(f"[structure] {self.steps} steps, receptive field {self.cfg.get('rf', '?')} px/step; block {self.Bc} coarse px "
            f"({self.Bc * self.cpp * self.cfg['mpp'] / 1000:.0f} km) + halo {self.Hc} -> each block computes a {n}x{n} px "
            f"window at {self.cfg['mpp']:.0f} m")

    def _block(self, bi, bj):
        with self._lock:                                         # one generation at a time (GPU + teacher coarse cache)
            key = (bi, bj)
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            t0 = time.perf_counter()
            Bc, Hc, f = self.Bc, self.Hc, self.cpp
            ci, cj = bi * Bc - Hc, bj * Bc - Hc
            c = self.src.coarse(ci, cj, ci + Bc + 2 * Hc, cj + Bc + 2 * Hc)
            z = self.stm.structure_generate(self.model, self.cfg, c, (ci, cj), self.seed, self.steps, self.dev, self.amp)
            z = np.ascontiguousarray(z[Hc * f:(Hc + Bc) * f, Hc * f:(Hc + Bc) * f])
            self._cache[key] = z
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
            self.block_ms.append((time.perf_counter() - t0) * 1000)
            return z

    def window(self, level, F, a0, b0, n, m):
        if abs(30.0 * 2 ** (F - level) - self.cfg["mpp"]) > 1e-6:
            raise ValueError(f"the structure model produces {self.cfg['mpp']:.0f} m/px; the cascade asked for level {level} "
                             f"of {F} ({30.0 * 2 ** (F - level):.0f} m/px). Use --student-start 0")
        B = self.Bc * self.cpp
        out = np.empty((n, m), np.float32)
        for bi in range(a0 // B, (a0 + n - 1) // B + 1):
            for bj in range(b0 // B, (b0 + m - 1) // B + 1):
                blk = self._block(bi, bj)
                ya, yb = max(a0, bi * B), min(a0 + n, (bi + 1) * B)
                xa, xb = max(b0, bj * B), min(b0 + m, (bj + 1) * B)
                out[ya - a0:yb - a0, xa - b0:xb - b0] = blk[ya - bi * B:yb - bi * B, xa - bj * B:xb - bj * B]
        return out


class StudentSource(TerrainSource):
    def __init__(self, ckpts=None, l0=None, start=0, block=512, halo=None, device=None, dtype=None,
                 seed=0, cache_blocks=64, gens=None, tag=None, log=print, reuse=None):
        self.log = log
        if gens is None:
            from eval_terrain import student_candidates
            gens = {}
            for p in ckpts:
                for g in student_candidates(p, device, seed, dtype, reuse=reuse):   # a shared checkpoint yields several levels
                    gens[g.cfg["level"]] = g
        self.gens = gens
        self.F = max(g.cfg["levels"] for g in gens.values()) - 1
        self.start = int(start)
        from student_model import stage_chain
        self.stages = stage_chain(gens, self.start, self.F)      # [(src, level, gen)], merged stages allowed
        if not self.stages:
            raise ValueError(f"nothing to generate: start level {self.start} is already the finest ({self.F})")
        need = [f"{s}->{L}" for s, L, _ in self.stages]
        if l0 is None:
            raise ValueError("a level-0 provider is required")
        self.l0 = l0
        f = 2 ** (self.F - self.start)
        margin = 0
        for s, L, g in self.stages:                             # border width spoiled by zero padding, in finest px
            margin = 2 ** (L - s) * margin + int(g.cfg.get("rf", 32))
        self.margin = margin
        self.halo = -(-(margin if halo is None else int(halo)) // f) * f
        if block % f:
            raise ValueError(f"block must be a multiple of {f} (finest px per start-level px)")
        self.B, self.f = int(block), f
        if tag is None:
            h = hashlib.sha1()
            for p in list(ckpts or []) + ([reuse] if reuse else []):
                st = os.stat(p)
                h.update(f"{os.path.abspath(p)}|{st.st_size}|{st.st_mtime_ns}".encode())
            tag = (f"student_{h.hexdigest()[:10]}_s{seed}_st{self.start}_{l0.tag}_B{self.B}_H{self.halo}{'_reuse' if reuse else ''}_"
                   f"{dtype or 'fp32'}")
        self.tag = tag
        self._cache = collections.OrderedDict()
        self._max = cache_blocks
        self._lock = threading.Lock()
        self._inflight = {}
        self._gpu = threading.Lock()
        self.block_ms = collections.deque(maxlen=20)
        self.blocks_made = 0
        self.log(f"[student] stages {', '.join(need)} from level {self.start}; block {self.B}px, halo {self.halo}px "
                 f"(margin {margin}px) -> each block computes a {self.B + 2 * self.halo}px finest-level window")

    # ------------------------------------------------------------ blocks
    def _gen(self, bi, bj):
        t0 = time.perf_counter()
        B, H, f = self.B, self.halo, self.f
        a0, b0 = (bi * B - H) // f, (bj * B - H) // f            # window origin in start-level pixels (exact)
        n = (B + 2 * H) // f
        z = self.l0.window(self.start, self.F, a0, b0, n, n).astype(np.float32)
        with self._gpu:
            for s, L, g in self.stages:
                o = 2 ** (s - self.start)                         # stage-input px per start-level px
                z = g(z, (a0 * o, b0 * o))                        # absolute origin -> position-hashed noise
        z = z[H:H + B, H:H + B]
        self.block_ms.append((time.perf_counter() - t0) * 1000)
        self.blocks_made += 1
        return np.clip(np.rint(z), -32000, 32000).astype(np.int16)

    def _block(self, bi, bj):
        key = (bi, bj)
        while True:
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                ev = self._inflight.get(key)
                mine = ev is None
                if mine:
                    ev = self._inflight[key] = threading.Event()
            if not mine:
                ev.wait()
                continue                                         # owner finished (or failed: we become the owner)
            try:
                arr = self._gen(bi, bj)
                with self._lock:
                    self._cache[key] = arr
                    while len(self._cache) > self._max:
                        self._cache.popitem(last=False)
                return arr
            finally:
                with self._lock:
                    self._inflight.pop(key, None)
                ev.set()

    def get(self, i1, j1, i2, j2):
        B = self.B
        out = np.empty((i2 - i1, j2 - j1), np.int16)
        for bi in range(i1 // B, (i2 - 1) // B + 1):
            for bj in range(j1 // B, (j2 - 1) // B + 1):
                blk = self._block(bi, bj)
                ya, yb = max(i1, bi * B), min(i2, (bi + 1) * B)
                xa, xb = max(j1, bj * B), min(j2, (bj + 1) * B)
                out[ya - i1:yb - i1, xa - j1:xb - j1] = blk[ya - bi * B:yb - bi * B, xa - bj * B:xb - bj * B]
        return out