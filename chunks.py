"""Chunk manager: keeps a circle of chunks around the player, generated in background threads.

Rings (radii in chunk units, measured from the player's chunk):
  active   (r_active)   : chunks the renderer meshes and draws
  prefetch (r_prefetch) : chunks whose heights are generated + cached in RAM (not drawn)
  evict    (r_evict)    : beyond this, heights are dropped from RAM (hysteresis > prefetch)
Every generated chunk is also written to disk, so "unloaded" chunks come back from disk
instead of the GPU. Generation is deterministic anyway, so the disk cache is purely a speed-up.

Chunk (ci, cj) covers pixel rows ci*N..ci*N+N and cols cj*N..cj*N+N INCLUSIVE (N+1 vertices,
sharing the border row/col with neighbours) plus a 1px apron for normals -> array of (N+3, N+3).
"""
import collections
import math
import os
import threading
import time

import numpy as np


def keys_within(center, r):
    ci, cj = center
    n = int(math.ceil(r))
    r2 = r * r
    return [(ci + a, cj + b) for a in range(-n, n + 1) for b in range(-n, n + 1) if a * a + b * b <= r2]


class ChunkManager:
    def __init__(self, source, chunk=128, r_active=6, r_prefetch=8, r_evict=10,
                 cache_dir=None, workers=2, log=print):
        self.src, self.N, self.log = source, chunk, log
        self.r_active, self.r_prefetch, self.r_evict = r_active, r_prefetch, r_evict
        self.cache_dir = None
        if cache_dir:
            self.cache_dir = os.path.join(cache_dir, f"{source.tag}_N{chunk}")
            os.makedirs(self.cache_dir, exist_ok=True)
        self._cv = threading.Condition()
        self._heights = {}
        self._disk = set()                                # keys with a cache file ("unloaded but stored")
        if self.cache_dir:
            for fn in os.listdir(self.cache_dir):
                if fn.startswith("c_") and fn.endswith(".npy"):
                    try:
                        _, a, b = fn[:-4].split("_")
                        self._disk.add((int(a), int(b)))
                    except ValueError:
                        pass
        self._pending, self._inflight, self._retry = set(), set(), {}
        self._center, self._heading = (0, 0), (0.0, 0.0)
        self._stop = False
        self.gen_ms = collections.deque(maxlen=20)
        self.generated = self.disk_hits = self.errors = self.consec_errors = 0
        self.last_error = None
        self._threads = [threading.Thread(target=self._worker, daemon=True) for _ in range(workers)]
        for t in self._threads:
            t.start()

    # ---------------- main-thread API ----------------
    def update(self, center, heading=(0.0, 0.0)):
        """Call whenever the player moves. Cheap. heading = (d_row, d_col) of travel/look."""
        ci, cj = center
        with self._cv:
            self._center, self._heading = center, heading
            want = keys_within(center, self.r_prefetch)
            r2e = self.r_evict ** 2
            for k in [k for k in self._heights if (k[0] - ci) ** 2 + (k[1] - cj) ** 2 > r2e]:
                del self._heights[k]                      # unload (stays on disk)
            now = time.time()
            self._pending = {k for k in want if k not in self._heights and k not in self._inflight
                             and self._retry.get(k, 0) <= now}
            self._cv.notify_all()

    def get(self, key):
        return self._heights.get(key)

    def wait_for(self, keys, on_progress=None):
        keys = list(keys)
        while True:
            have = sum(1 for k in keys if k in self._heights)
            if on_progress:
                on_progress(have, len(keys))
            if have == len(keys):
                return
            if self.consec_errors >= 5:
                raise RuntimeError(f"terrain generation failing: {self.last_error}")
            time.sleep(0.03)

    def height_at(self, i, j):
        """Bilinear height (metres) at fractional pixel coords, or None if chunk not loaded."""
        N = self.N
        ci, cj = int(math.floor(i / N)), int(math.floor(j / N))
        a = self._heights.get((ci, cj))
        if a is None:
            return None
        fi, fj = i - (ci * N - 1), j - (cj * N - 1)
        i0, j0 = int(fi), int(fj)
        ti, tj = fi - i0, fj - j0
        top = a[i0, j0] * (1 - tj) + a[i0, j0 + 1] * tj
        bot = a[i0 + 1, j0] * (1 - tj) + a[i0 + 1, j0 + 1] * tj
        return float(top * (1 - ti) + bot * ti)

    def stats(self):
        with self._cv:
            ms = sum(self.gen_ms) / len(self.gen_ms) if self.gen_ms else 0.0
            return dict(cached=len(self._heights), pending=len(self._pending),
                        inflight=len(self._inflight), generated=self.generated,
                        disk_hits=self.disk_hits, errors=self.errors, avg_ms=ms)

    def snapshot(self):
        """Chunk states for the minimap: sets of keys in RAM / queued / being generated / stored on disk."""
        with self._cv:
            return dict(ram=set(self._heights), pending=set(self._pending),
                        inflight=set(self._inflight), disk=set(self._disk))

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify_all()

    # ---------------- workers ----------------
    def _pick(self):
        ci, cj = self._center
        hi, hj = self._heading
        hn = math.hypot(hi, hj)

        def score(k):
            di, dj = k[0] - ci, k[1] - cj
            d = math.hypot(di, dj)
            s = d
            if hn > 1e-6 and d > 0:
                s -= (di * hi + dj * hj) / (d * hn)       # prefer chunks ahead of travel
            if d <= self.r_active:
                s -= 100.0                                 # visible ring always first
            return s
        return min(self._pending, key=score)

    def _worker(self):
        while True:
            with self._cv:
                while not self._stop and not self._pending:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                key = self._pick()
                self._pending.discard(key)
                self._inflight.add(key)
            try:
                t0 = time.time()
                arr, from_disk = self._load_or_generate(key)
                dt = (time.time() - t0) * 1000
            except Exception as e:
                with self._cv:
                    self._inflight.discard(key)
                    self._retry[key] = time.time() + 5
                    self.errors += 1
                    self.consec_errors += 1
                    self.last_error = repr(e)
                self.log(f"[chunks] {key} failed: {e!r}")
                time.sleep(0.5)
                continue
            with self._cv:
                self._inflight.discard(key)
                self.consec_errors = 0
                if from_disk:
                    self.disk_hits += 1
                else:
                    self.generated += 1
                    self.gen_ms.append(dt)
                ci, cj = self._center
                if (key[0] - ci) ** 2 + (key[1] - cj) ** 2 <= self.r_evict ** 2:
                    self._heights[key] = arr
                self._cv.notify_all()

    def _path(self, key):
        return os.path.join(self.cache_dir, f"c_{key[0]}_{key[1]}.npy") if self.cache_dir else None

    def _load_or_generate(self, key):
        N, p = self.N, self._path(key)
        if p and os.path.exists(p):
            try:
                a = np.load(p)
                if a.shape == (N + 3, N + 3):
                    return a, True
            except Exception:
                pass                                       # corrupt -> regenerate
        ci, cj = key
        a = self.src.get(ci * N - 1, cj * N - 1, ci * N + N + 2, cj * N + N + 2)
        if a.shape != (N + 3, N + 3):
            raise ValueError(f"source returned {a.shape}")
        if p:
            tmp = p + f".{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                np.save(f, a)
            os.replace(tmp, p)
            with self._cv:
                self._disk.add(key)
        return a, False
