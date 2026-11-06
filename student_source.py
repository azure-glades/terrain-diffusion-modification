"""Student cascade as an explorer terrain source.

Stateless and random-access by construction: the world is cut into fixed BLOCKs (default 512 px at the finest level).
A block is a pure function of (level-0 content, noise seed, block index): the cascade runs on a window padded by a halo
of >= the cascade's receptive-field margin, then the halo is cropped. Chunk requests are assembled from blocks, so
neighbouring chunks always agree and a block is generated once (LRU-cached) however many chunks need it.

Level-0 providers (anything with .window(level, F, a0, b0, n, m) -> float32 metres at level `level`):
  ProceduralL0 : fractal noise in the DemoSource style, point-sampled at 480 m/px (no model, no server)
  TeacherL0    : any TerrainSource (the real teacher via HttpSource, or DemoSource) block-averaged down to `level`
                 -> a hybrid: the teacher supplies the coarse structure, the students add the detail
"""
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
        need = list(range(self.start + 1, self.F + 1))
        if not need or any(L not in gens for L in need):
            raise ValueError(f"need a student for every level {need}; have {sorted(gens)}")
        if l0 is None:
            raise ValueError("a level-0 provider is required")
        self.l0 = l0
        f = 2 ** (self.F - self.start)
        margin = 0
        for L in need:                                          # border width spoiled by zero padding, in finest px
            margin = 2 * margin + int(gens[L].cfg.get("rf", 32))
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
        self.log(f"[student] levels {need} from level {self.start}; block {self.B}px, halo {self.halo}px "
                 f"(margin {margin}px) -> each block computes a {self.B + 2 * self.halo}px finest-level window")

    # ------------------------------------------------------------ blocks
    def _gen(self, bi, bj):
        t0 = time.perf_counter()
        B, H, f = self.B, self.halo, self.f
        a0, b0 = (bi * B - H) // f, (bj * B - H) // f            # window origin in start-level pixels (exact)
        n = (B + 2 * H) // f
        z = self.l0.window(self.start, self.F, a0, b0, n, n).astype(np.float32)
        with self._gpu:
            for L in range(self.start + 1, self.F + 1):
                o = 2 ** (L - 1 - self.start)                     # level-(L-1) px per start-level px
                z = self.gens[L](z, (a0 * o, b0 * o))             # absolute origin -> position-hashed noise
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