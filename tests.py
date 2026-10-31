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
        time.sleep(0.5)
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
        shutil.rmtree(d)

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

if __name__ == "__main__":
    test_seams_and_determinism(); test_mesh(); test_streaming(); test_http_client(); print("ALL PASSED")
