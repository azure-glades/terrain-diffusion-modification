"""Headless tests: python tests.py   (no GPU / model / display needed)"""
import http.server, os, shutil, tempfile, threading, time
import numpy as np
import meshing
from chunks import ChunkManager, keys_within
from sources import DemoSource, HttpSource

N = 32

def test_seams_and_determinism():
    s = DemoSource(3)
    a = s.get(-5, -5, 40, 40); b = s.get(10, 10, 20, 20)
    assert (a[15:25, 15:25] == b).all()                 # overlapping windows agree
    assert (DemoSource(3).get(0, 0, 64, 64) == DemoSource(3).get(0, 0, 64, 64)).all()
    assert not (DemoSource(3).get(0, 0, 64, 64) == DemoSource(4).get(0, 0, 64, 64)).all()
    print("ok seams/determinism")

def test_mesh():
    h = DemoSource(1).get(-1, -1, N + 2, N + 2)
    v = meshing.build_vertices(h, N)
    assert v.shape == ((N + 1) ** 2,) and v.dtype.itemsize == 28
    idx = meshing.make_indices(N)
    assert idx.max() == (N + 1) ** 2 - 1 and len(idx) == N * N * 6
    # adjacent chunks must share identical border vertices (heights, normals) -> no cracks/lighting seams
    ha = DemoSource(1).get(-1, -1, N + 2, N + 2)
    hb = DemoSource(1).get(-1, N - 1, N + 2, 2 * N + 2)
    va = meshing.build_vertices(ha, N).reshape(N + 1, N + 1); vb = meshing.build_vertices(hb, N).reshape(N + 1, N + 1)
    assert (va["v"][:, N, 2] == vb["v"][:, 0, 2]).all()
    assert np.allclose(va["n"][:, N], vb["n"][:, 0], atol=1e-6)
    assert (va["c"][:, N] == vb["c"][:, 0]).all()
    print("ok mesh + seam-identical borders")

def test_streaming():
    d = tempfile.mkdtemp()
    try:
        m = ChunkManager(DemoSource(7), chunk=N, r_active=3, r_prefetch=5, r_evict=7, cache_dir=d, workers=2, log=lambda *a: None)
        m.update((0, 0), (0, 1))
        m.wait_for(keys_within((0, 0), 3))
        m.wait_for(keys_within((0, 0), 5))                  # prefetch ring fills in the background
        assert all(m.get(k) is not None for k in keys_within((0, 0), 5)), "prefetch ring incomplete"
        n0 = m.stats()["cached"]
        # walk east 30 chunks, maintaining the ring
        for step in range(1, 31):
            m.update((0, step), (0, 1))
            m.wait_for(keys_within((0, step), 3))
            assert m.stats()["cached"] <= len(keys_within((0, step), 7)), "RAM not bounded"
        time.sleep(0.5)
        assert m.get((0, 0)) is None, "far chunk should be evicted from RAM"
        gen_before = m.stats()["generated"]
        m.update((0, 0), (0, -1)); m.wait_for(keys_within((0, 0), 3))      # come back
        s = m.stats()
        assert s["disk_hits"] > 0 and s["generated"] <= gen_before + 5, s   # served from disk, not regenerated
        assert m.height_at(0.5, 0.5) is not None and m.height_at(1e6, 1e6) is None
        m.close()
        print("ok streaming: ring maintained, RAM bounded, evicted chunks reloaded from disk", s)
    finally:
        time.sleep(0.2); shutil.rmtree(d, ignore_errors=True)

def test_http_client():
    src = DemoSource(5)
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200); self.end_headers(); self.wfile.write(b"ok"); return
            q = dict(p.split("=") for p in self.path.split("?")[1].split("&"))
            i1, j1, i2, j2 = (int(q[k]) for k in ("i1", "j1", "i2", "j2"))
            e = src.get(i1, j1, i2, j2); hh, ww = e.shape
            body = e.astype("<i2").tobytes() + np.zeros((hh, ww, 4), "<f4").tobytes()   # documented layout
            self.send_response(200); self.send_header("X-Height", str(hh)); self.send_header("X-Width", str(ww))
            self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    c = HttpSource(f"http://127.0.0.1:{srv.server_address[1]}")
    assert c.wait_ready(5)
    assert (c.get(3, 4, 40, 50) == src.get(3, 4, 40, 50)).all()
    srv.shutdown()
    print("ok http client parses documented payload")

def _fake_gens(F=4, rf=6):
    """Local-only fake students (bilinear up2 + a 3x3 smoothing detail term) so StudentSource logic runs without torch."""
    from make_pyramid_data import up2
    def make(L):
        def g(c, origin=(0, 0)):
            z = up2(c.astype(np.float32))
            d = z - 0.25 * (np.roll(z, 1, 0) + np.roll(z, -1, 0) + np.roll(z, 1, 1) + np.roll(z, -1, 1))
            return (z + 3.0 * d).astype(np.float32)
        g.cfg = dict(level=L, levels=F + 1, rf=rf)
        return g
    return {L: make(L) for L in range(1, F + 1)}

def test_student_source():
    from student_source import ProceduralL0, StudentSource, TeacherL0
    l0 = ProceduralL0(2)
    a = StudentSource(l0=l0, gens=_fake_gens(), block=128, log=lambda *x: None, tag="fa")
    b = StudentSource(l0=l0, gens=_fake_gens(), block=256, log=lambda *x: None, tag="fb")   # different block grid
    wa, wb = a.get(-70, 90, 190, 300), b.get(-70, 90, 190, 300)
    assert wa.shape == (260, 210) and np.abs(wa.astype(int) - wb).max() <= 1, np.abs(wa.astype(int) - wb).max()
    # chunk windows (N+3, overlapping by construction) agree with each other across a block border
    c1, c2 = a.get(-1, 125, 132, 258), a.get(-1, 253, 132, 386)
    assert (c1[:, 128:] == c2[:, :5]).all()
    assert (a.get(5, 5, 40, 40) == a.get(5, 5, 40, 40)).all()                                # deterministic / cached
    assert a.blocks_made > 0 and a.margin > 0
    h = StudentSource(l0=TeacherL0(DemoSource(1)), gens=_fake_gens(), start=2, block=128, log=lambda *x: None, tag="ft")
    assert h.get(0, 0, 100, 100).shape == (100, 100)
    print("ok student source: block-size independent, seamless across blocks, teacher hybrid works, margin", a.margin)

def test_snapshot():
    d = tempfile.mkdtemp()
    try:
        m = ChunkManager(DemoSource(9), chunk=N, r_active=2, r_prefetch=3, r_evict=4, cache_dir=d, workers=2, log=lambda *a: None)
        m.update((0, 0)); m.wait_for(keys_within((0, 0), 2)); time.sleep(0.4)
        s = m.snapshot()
        assert set(keys_within((0, 0), 3)) <= s["ram"] and s["ram"] <= s["disk"]
        m.update((0, 40)); time.sleep(0.2)                       # far away: old ones leave RAM but stay on disk
        s = m.snapshot()
        assert (0, 0) not in s["ram"] and (0, 0) in s["disk"]
        m.close()
        m2 = ChunkManager(DemoSource(9), chunk=N, cache_dir=d, workers=1, log=lambda *a: None)
        assert (0, 0) in m2.snapshot()["disk"]                   # disk index rebuilt on restart
        m2.close()
        print("ok snapshot: ram / pending / inflight / disk states")
    finally:
        time.sleep(0.2); shutil.rmtree(d, ignore_errors=True)

if __name__ == "__main__":
    test_seams_and_determinism(); test_mesh(); test_streaming(); test_http_client(); test_student_source(); test_snapshot()
    print("ALL PASSED")
