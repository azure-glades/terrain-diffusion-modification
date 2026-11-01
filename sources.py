"""Terrain sources: anything that can return an int16 elevation window in metres.

get(i1, j1, i2, j2) -> int16 array of shape (i2-i1, j2-j1).
(i = row / north-south axis, j = column / east-west axis, i2/j2 exclusive.)
Sources must be deterministic and seamless: overlapping windows agree.
"""
import time
import urllib.error
import urllib.request

import numpy as np


class TerrainSource:
    tag = "base"

    def get(self, i1, j1, i2, j2):
        raise NotImplementedError


class HttpSource(TerrainSource):
    """Client for the repo's API server (`python -m terrain_diffusion api <model>`).

    GET /terrain?i1&j1&i2&j2&scale -> headers X-Height / X-Width,
    body = int16 elevation (H*W) followed by float32 climate (H*W*4, unused here).
    """

    def __init__(self, url="http://127.0.0.1:8000", scale=1, seed=0, model="30m",
                 timeout=900, retries=2, variant=""):
        self.url = url.rstrip("/")
        self.scale = scale
        self.timeout = timeout
        self.retries = retries
        # variant keeps differently-configured servers (e.g. one-step latent) in separate disk caches
        self.tag = f"http_{model}_s{seed}_x{scale}{variant}"

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3) as r:
                return r.status == 200
        except Exception:
            return False

    def wait_ready(self, timeout=600, alive=lambda: True, log=print):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.healthy():
                return True
            if not alive():
                raise RuntimeError("terrain server process exited early")
            time.sleep(1.0)
        return False

    def get(self, i1, j1, i2, j2):
        q = f"{self.url}/terrain?i1={i1}&j1={j1}&i2={i2}&j2={j2}&scale={self.scale}"
        h, w = i2 - i1, j2 - j1
        last = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(q, timeout=self.timeout) as r:
                    hh = int(r.headers.get("X-Height", h))
                    ww = int(r.headers.get("X-Width", w))
                    body = r.read()
                if (hh, ww) != (h, w):
                    raise ValueError(f"server returned {hh}x{ww}, expected {h}x{w}")
                if len(body) < h * w * 2:
                    raise ValueError(f"short body: {len(body)} bytes for {h}x{w}")
                return np.frombuffer(body, dtype="<i2", count=h * w).reshape(h, w).astype(np.int16)
            except urllib.error.HTTPError as e:
                last = RuntimeError(f"HTTP {e.code}: {e.read()[:200]!r}")
            except Exception as e:  # network hiccup / server busy
                last = e
            time.sleep(0.5)
        raise last


class DemoSource(TerrainSource):
    """Procedural stand-in (hash-based fractal noise). No GPU / model needed.
    Good for testing the chunk manager and renderer. Deterministic + seamless."""

    def __init__(self, seed=0):
        self.seed = int(seed)
        self.tag = f"demo_s{self.seed}"

    def _hash(self, ix, iy, salt):
        with np.errstate(over="ignore"):
            h = (ix.astype(np.uint64) * np.uint64(374761393)
                 + iy.astype(np.uint64) * np.uint64(668265263)
                 + np.uint64(self.seed * 7919 + salt) * np.uint64(2246822519))
            h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
            h = h ^ (h >> np.uint64(16))
        return (h & np.uint64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)

    def _vnoise(self, x, y, salt):
        ix, iy = np.floor(x), np.floor(y)
        fx, fy = x - ix, y - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        ix, iy = ix.astype(np.int64), iy.astype(np.int64)
        a, b = self._hash(ix, iy, salt), self._hash(ix + 1, iy, salt)
        c, d = self._hash(ix, iy + 1, salt), self._hash(ix + 1, iy + 1, salt)
        return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fy

    def _fbm(self, x, y, wavelength, octaves, salt, ridged=False):
        total, amp, norm = 0.0, 1.0, 0.0
        for o in range(octaves):
            n = self._vnoise(x / wavelength, y / wavelength, salt + o * 31)
            if ridged:
                n = 1.0 - np.abs(2 * n - 1)
                n = n * n
            total += n * amp
            norm += amp
            amp *= 0.5
            wavelength *= 0.5
        return total / norm

    def sample(self, ii, jj):
        """Float elevation (metres) on the grid of 1-D coordinate arrays ii (rows) x jj (cols); fractional coords allowed."""
        ii, jj = np.meshgrid(np.asarray(ii, dtype=np.float64), np.asarray(jj, dtype=np.float64), indexing="ij")
        cont = self._fbm(jj, ii, 1500.0, 4, 1)               # continents
        land = np.clip((cont - 0.47) * 6.0, 0.0, 1.0)         # 0 = ocean
        ridge = self._fbm(jj, ii, 400.0, 6, 100, ridged=True)
        rough = self._fbm(jj, ii, 60.0, 4, 200)
        h = land * (80 + 5200 * ridge ** 1.3 * land ** 0.5 + 300 * rough)
        return np.where(cont < 0.47, -20.0 * (0.47 - cont) * 10, h)

    def get(self, i1, j1, i2, j2):
        h = self.sample(np.arange(i1, i2), np.arange(j1, j2))
        return np.clip(np.rint(h), -32000, 32000).astype(np.int16)
