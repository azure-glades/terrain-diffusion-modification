"""Heightmap -> vertex buffer (numpy only; no Panda3D dependency so it is easy to test).

World convention: x = column (j), y = -row (i)  (north is +y, no mirroring), z = height.
One world unit = one heightmap pixel horizontally; z = metres / mpp * vexag.
Vertices are local to the chunk (origin at its NW corner); the node is positioned at (cj*N, -ci*N).
"""
import numpy as np

VERTEX_DTYPE = np.dtype([("v", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 4)])  # == Panda V3n3c4

_STOPS_E = np.array([0, 4, 40, 400, 1000, 1800, 2700, 3600, 4300], dtype=np.float32)
_STOPS_C = np.array([
    (0.80, 0.75, 0.55),   # beach
    (0.45, 0.58, 0.28),   # lowland
    (0.33, 0.52, 0.24),   # grass
    (0.22, 0.40, 0.20),   # forest
    (0.42, 0.40, 0.30),   # scrub/rock
    (0.50, 0.47, 0.42),   # alpine rock
    (0.62, 0.60, 0.58),   # high rock
    (0.93, 0.94, 0.97),   # snow
    (0.98, 0.98, 1.00),
], dtype=np.float32)
ROCK = np.array((0.40, 0.38, 0.36), dtype=np.float32)
WATER = np.array((0.08, 0.24, 0.42), dtype=np.float32)
SUN = np.array((-0.55, -0.35, 0.76), dtype=np.float32)
SUN /= np.linalg.norm(SUN)


def make_indices(N):
    """Triangle indices for an (N+1)x(N+1) vertex grid, CCW seen from above (+z)."""
    w = N + 1
    r, c = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    v00 = (r * w + c).ravel()
    v01, v10, v11 = v00 + 1, v00 + w, v00 + w + 1
    idx = np.stack([v00, v10, v01, v01, v10, v11], axis=1).ravel()
    return idx.astype(np.uint16 if w * w < 65536 else np.uint32)


def build_vertices(h_pad, N, mpp=30.0, vexag=1.5):
    """h_pad: (N+3, N+3) int16 metres (1px apron). Returns structured array of (N+1)^2 vertices."""
    h = h_pad.astype(np.float32)
    ocean = h <= 0.0
    z = np.maximum(h, 0.0) / mpp * vexag

    gx = (z[1:-1, 2:] - z[1:-1, :-2]) * 0.5            # dz/dx   (x = +col)
    gi = (z[2:, 1:-1] - z[:-2, 1:-1]) * 0.5            # dz/drow (y = -row, so dz/dy = -gi)
    nrm = np.stack([-gx, gi, np.ones_like(gx)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    e = h[1:-1, 1:-1]
    oc = ocean[1:-1, 1:-1]
    col = np.stack([np.interp(e, _STOPS_E, _STOPS_C[:, k]) for k in range(3)], axis=-1)
    steep = np.clip(((1.0 - nrm[..., 2]) - 0.10) / 0.25, 0.0, 1.0)
    steep = steep * steep * (3 - 2 * steep)
    col = col * (1 - 0.8 * steep[..., None]) + ROCK * (0.8 * steep[..., None])
    lam = np.clip(nrm @ SUN, 0.0, 1.0)
    col = col * (0.30 + 0.85 * lam[..., None])
    wcol = WATER * (0.85 + 0.15 * lam[..., None])
    col = np.where(oc[..., None], wcol, col)
    nrm = np.where(oc[..., None], np.array([0, 0, 1], np.float32), nrm)

    W = N + 1
    out = np.empty((W, W), dtype=VERTEX_DTYPE)
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="xy")
    out["v"][..., 0] = jj
    out["v"][..., 1] = -ii
    out["v"][..., 2] = z[1:-1, 1:-1]
    out["n"] = nrm
    out["c"][..., :3] = np.clip(col * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out["c"][..., 3] = 255
    return out.reshape(-1)
