"""Panda3D explorer: fly camera + streaming terrain meshes driven by ChunkManager."""
import math
import time

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (AntialiasAttrib, CardMaker, Fog, Geom, GeomEnums, GeomNode, GeomTriangles,
                          GeomVertexData, GeomVertexFormat, KeyboardButton, SamplerState, TextNode,
                          Texture, TransparencyAttrib, WindowProperties)

import meshing
from chunks import keys_within

SKY = (0.62, 0.76, 0.90)

# minimap legend: state -> (label, RGBA). Priority when several apply: drawn > RAM > generating > queued > disk > none.
MM_STATES = [
    ("drawn", "drawn (loaded)", (70, 200, 90, 255)),
    ("ram", "loaded, not drawn", (70, 130, 220, 255)),
    ("gen", "being generated", (255, 150, 30, 255)),
    ("queued", "queued", (235, 215, 90, 255)),
    ("disk", "unloaded (on disk)", (105, 105, 118, 255)),
    ("none", "not generated", (24, 26, 32, 255)),
]


class Minimap:
    """Top-down chunk map in the corner: one cell per chunk, north up (row index grows downward on screen),
    drawn into a numpy RGBA buffer and uploaded as a texture ~10x per second."""

    def __init__(self, app, size=0.56):
        mgr = app.mgr
        self.app, self.mgr = app, mgr
        self.R = int(math.ceil(mgr.r_evict)) + 1
        self.side = 2 * self.R + 1
        self.cs = max(4, min(14, 230 // self.side))
        self.px = self.side * self.cs
        self.col = {k: np.array(c, np.uint8) for k, _, c in MM_STATES}
        self.tex = Texture("minimap")
        self.tex.setup2dTexture(self.px, self.px, Texture.T_unsigned_byte, Texture.F_rgba8)
        self.tex.setMagfilter(SamplerState.FT_nearest)
        self.tex.setMinfilter(SamplerState.FT_nearest)
        self.root = app.a2dTopRight.attachNewNode("minimap")
        cm = CardMaker("mm")
        cm.setFrame(0, 1, 0, 1)
        self.card = self.root.attachNewNode(cm.generate())
        self.card.setScale(size)
        self.card.setPos(-size - 0.03, 0, -size - 0.03)
        self.card.setTexture(self.tex)
        self.card.setTransparency(TransparencyAttrib.MAlpha)
        self.counts = {}
        self.labels = {}
        y = -size - 0.03 - 0.055
        for key, label, rgba in MM_STATES:
            sw = CardMaker("sw")
            sw.setFrame(0, 0.032, 0, 0.032)
            n = self.root.attachNewNode(sw.generate())
            n.setColor(*(v / 255 for v in rgba))
            n.setPos(-size - 0.03, 0, y - 0.006)
            t = OnscreenText(text="", parent=self.root, pos=(-size - 0.03 + 0.045, y), scale=0.036,
                             align=TextNode.ALeft, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), mayChange=True)
            self.labels[key] = (t, label)
            y -= 0.047
        self.shown = True

    def toggle(self):
        self.shown = not self.shown
        (self.root.show if self.shown else self.root.hide)()

    def update(self, center, drawn, pos_ij, heading_deg):
        if not self.shown:
            return
        R, side, cs = self.R, self.side, self.cs
        snap = self.mgr.snapshot()
        ci, cj = center
        cells = np.empty((side, side, 4), np.uint8)
        counts = dict.fromkeys(self.col, 0)
        for a in range(side):
            for b in range(side):
                k = (ci - R + a, cj - R + b)
                if k in drawn:
                    s = "drawn"
                elif k in snap["ram"]:
                    s = "ram"
                elif k in snap["inflight"]:
                    s = "gen"
                elif k in snap["pending"]:
                    s = "queued"
                elif k in snap["disk"]:
                    s = "disk"
                else:
                    s = "none"
                cells[a, b] = self.col[s]
                counts[s] += 1
        img = np.repeat(np.repeat(cells, cs, axis=0), cs, axis=1)
        edge = np.zeros(img.shape[:2], bool)                      # darker grid lines
        edge[::cs, :] = True
        edge[:, ::cs] = True
        img[edge] = (img[edge] * 0.62).astype(np.uint8)
        yy, xx = np.mgrid[0:self.px, 0:self.px]
        c0 = (R + 0.5) * cs
        dist = np.hypot(yy - c0, xx - c0)
        for r, rgba in ((self.mgr.r_active, (255, 255, 255, 255)), (self.mgr.r_prefetch, (120, 220, 255, 255))):
            ring = np.abs(dist - (r + 0.5) * cs) < 0.75
            img[ring] = rgba
        # player + heading (map: right = +col, up = +world y = -row)
        N = self.app.N
        py = (pos_ij[0] / N - (ci - R)) * cs
        px = (pos_ij[1] / N - (cj - R)) * cs
        h = math.radians(heading_deg)
        dx, dy = -math.sin(h), -math.cos(h)
        for t in np.linspace(0, 2.4 * cs, int(2.4 * cs * 2)):
            self._dot(img, py + dy * t, px + dx * t, (255, 60, 60, 255), 1)
        self._dot(img, py, px, (255, 255, 255, 255), 2)
        img[0, :], img[-1, :], img[:, 0], img[:, -1] = 255, 255, 255, 255
        self.tex.setRamImageAs(np.ascontiguousarray(img[::-1]).tobytes(), "RGBA")   # panda rows start at the bottom
        for key, (t, label) in self.labels.items():
            t.setText(f"{label}: {counts[key]}")

    @staticmethod
    def _dot(img, y, x, rgba, r):
        y0, x0 = int(round(y)), int(round(x))
        img[max(y0 - r, 0):max(y0 + r + 1, 0), max(x0 - r, 0):max(x0 + r + 1, 0)] = rgba


class Explorer(ShowBase):
    def __init__(self, mgr, mpp=30.0, vexag=1.5, speed=60.0, start=(0, 0), clearance=1.5):
        super().__init__()
        self.mgr, self.N, self.mpp, self.vexag = mgr, mgr.N, mpp, vexag
        self.speed, self.clearance = speed, clearance
        self.mesh_ver = 0
        self.meshes = {}                 # key -> (NodePath, version)
        self.build_queue = []
        self.hud_on, self.captured = True, False
        self.sens = 0.12
        self.heading, self.pitch = 0.0, -15.0

        self.setBackgroundColor(*SKY, 1.0)
        self.disableMouse()
        self.render.setAntialias(AntialiasAttrib.MAuto)
        far = (mgr.r_active + 1.5) * self.N * 1.2
        self.camLens.setNearFar(0.3, far)
        self.camLens.setFov(75)
        fog = Fog("fog")
        fog.setColor(*SKY)
        fog.setLinearRange(mgr.r_active * self.N * 0.55, mgr.r_active * self.N * 0.98)
        self.render.setFog(fog)

        self.terrain = self.render.attachNewNode("terrain")
        self.prim = GeomTriangles(Geom.UHStatic)
        idx = meshing.make_indices(self.N)
        self.prim.setIndexType(GeomEnums.NT_uint16 if idx.dtype.itemsize == 2 else GeomEnums.NT_uint32)
        arr = self.prim.modifyVertices()
        arr.uncleanSetNumRows(len(idx))
        arr.modifyHandle().copyDataFrom(idx.tobytes())

        self.hud = OnscreenText(text="", parent=self.a2dTopLeft, pos=(0.04, -0.07), scale=0.045,
                                align=TextNode.ALeft, fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1), mayChange=True)
        self.help = OnscreenText(
            text="WASD move | Space/E up, Q/C down | Shift fast | wheel speed | -/= vert. exaggeration\n"
                 "Tab mouse capture | F wireframe | P screenshot | H hide HUD | M minimap | Esc quit",
            parent=self.a2dBottomLeft, pos=(0.04, 0.09), scale=0.04, align=TextNode.ALeft,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

        self.minimap = Minimap(self)

        # ----- initial area (blocking, with progress) -----
        si, sj = start
        self.pos = [float(sj), -float(si), 0.0]
        c0 = self._chunk_of_pos()
        mgr.update(c0)
        ring = keys_within(c0, mgr.pregen_radius)
        t0 = time.time()

        def progress(have, total):
            self.hud.setText(f"Generating terrain around start: {have}/{total} chunks")
            self._minimap_update()
            self.taskMgr.step()
        mgr.wait_for(ring, progress)
        for k in sorted(ring, key=lambda k: math.hypot(k[0] - c0[0], k[1] - c0[1])):
            self._build(k)
        g = mgr.height_at(-self.pos[1], self.pos[0]) or 0.0
        self.pos[2] = max(g, 0.0) / mpp * vexag + 25.0
        print(f"[app] start area ready: {len(ring)} chunks in {time.time() - t0:.1f}s")

        self.accept("escape", self.userExit)
        self.accept("tab", self._toggle_capture)
        self.accept("wheel_up", self._speed, [1.25])
        self.accept("wheel_down", self._speed, [0.8])
        self.accept("-", self._exag, [0.8])
        self.accept("=", self._exag, [1.25])
        self.accept("f", self._wire)
        self.accept("h", self._toggle_hud)
        self.accept("p", self._shot)
        self.accept("m", self.minimap.toggle)
        self.wire = False
        self._set_capture(True)

        self.taskMgr.add(self._move_task, "move")
        self.taskMgr.add(self._stream_task, "stream")
        self.taskMgr.add(self._build_task, "build")
        self.taskMgr.doMethodLater(0.25, self._hud_task, "hud")
        self.taskMgr.doMethodLater(0.1, self._minimap_task, "minimap")
        self.last_tick = 0.0
        self.fps_frames = 0

    # ----------------------------------------------------------- helpers
    def _chunk_of_pos(self):
        i, j = -self.pos[1], self.pos[0]
        return (int(math.floor(i / self.N)), int(math.floor(j / self.N)))

    def _minimap_update(self):
        self.minimap.update(self._chunk_of_pos(), self.meshes, (-self.pos[1], self.pos[0]), self.heading)

    def _minimap_task(self, task):
        self._minimap_update()
        return task.again

    def _speed(self, f):
        self.speed = min(max(self.speed * f, 2.0), 4000.0)

    def _exag(self, f):
        self.vexag = min(max(self.vexag * f, 0.25), 12.0)
        self.mesh_ver += 1

    def _wire(self):
        self.wire = not self.wire
        self.terrain.setRenderModeWireframe() if self.wire else self.terrain.setRenderModeFilled()

    def _toggle_hud(self):
        self.hud_on = not self.hud_on
        (self.hud.show if self.hud_on else self.hud.hide)()
        (self.help.show if self.hud_on else self.help.hide)()

    def _shot(self):
        name = time.strftime("terrain_%H%M%S.png")
        self.screenshot(name, defaultFilename=False)
        print("[app] saved", name)

    def _toggle_capture(self):
        self._set_capture(not self.captured)

    def _set_capture(self, on):
        self.captured = on
        if not hasattr(self.win, "requestProperties"):      # offscreen buffer (self-test)
            self.captured = False
            return
        p = WindowProperties()
        p.setCursorHidden(on)
        self.win.requestProperties(p)
        if on and self.win.getXSize():
            self.win.movePointer(0, self.win.getXSize() // 2, self.win.getYSize() // 2)

    # ----------------------------------------------------------- per-frame
    def _down(self, ch):
        mw = self.mouseWatcherNode
        return mw is not None and mw.isButtonDown(KeyboardButton.asciiKey(ch))

    def _btn(self, button):
        mw = self.mouseWatcherNode
        return mw is not None and mw.isButtonDown(button)

    def _move_task(self, task):
        dt = min(globalClock.getDt(), 0.1)
        if self.captured and self.mouseWatcherNode is not None and self.mouseWatcherNode.hasMouse():
            cx, cy = self.win.getXSize() // 2, self.win.getYSize() // 2
            md = self.win.getPointer(0)
            dx, dy = md.getX() - cx, md.getY() - cy
            if (dx or dy) and self.win.movePointer(0, cx, cy):
                self.heading -= dx * self.sens
                self.pitch = max(-89.0, min(89.0, self.pitch - dy * self.sens))
        self.camera.setHpr(self.heading, self.pitch, 0)

        v = self.speed * (4.0 if self._btn(KeyboardButton.shift()) else 1.0) * dt
        q = self.camera.getQuat(self.render)
        fwd, right = q.getForward(), q.getRight()
        mv = [0.0, 0.0, 0.0]
        if self._down("w"):
            mv = [mv[k] + fwd[k] for k in range(3)]
        if self._down("s"):
            mv = [mv[k] - fwd[k] for k in range(3)]
        if self._down("d"):
            mv = [mv[k] + right[k] for k in range(3)]
        if self._down("a"):
            mv = [mv[k] - right[k] for k in range(3)]
        if self._down("e") or self._btn(KeyboardButton.space()):
            mv[2] += 1.0
        if self._down("q") or self._down("c"):
            mv[2] -= 1.0
        for k in range(3):
            self.pos[k] += mv[k] * v

        g = self.mgr.height_at(-self.pos[1], self.pos[0])        # keep above the ground
        if g is not None:
            self.pos[2] = max(self.pos[2], max(g, 0.0) / self.mpp * self.vexag + self.clearance)
        self.camera.setPos(*self.pos)
        self.fps_frames += 1
        return task.cont

    def _stream_task(self, task):
        """Keeps the ring maintained; ~10 Hz."""
        now = globalClock.getFrameTime()
        if now - self.last_tick < 0.1:
            return task.cont
        self.last_tick = now
        c = self._chunk_of_pos()
        fwd = self.camera.getQuat(self.render).getForward()
        heading = (-fwd[1], fwd[0])                                  # (d_row, d_col)
        self.mgr.update(c, heading)

        active = keys_within(c, self.mgr.r_active)
        want = set(active)
        todo = []
        for k in active:
            cur = self.meshes.get(k)
            if (cur is None or cur[1] != self.mesh_ver) and self.mgr.get(k) is not None:
                todo.append(k)
        todo.sort(key=lambda k: (k[0] - c[0]) ** 2 + (k[1] - c[1]) ** 2)
        self.build_queue = todo
        # unload meshes outside active radius + 1 (hysteresis)
        r2 = (self.mgr.r_active + 1.0) ** 2
        for k in [k for k in self.meshes if (k[0] - c[0]) ** 2 + (k[1] - c[1]) ** 2 > r2]:
            self.meshes.pop(k)[0].removeNode()
        return task.cont

    def _build_task(self, task):
        t0 = time.perf_counter()
        while self.build_queue and (time.perf_counter() - t0) < 0.006:
            self._build(self.build_queue.pop(0))
        return task.cont

    def _build(self, key):
        h = self.mgr.get(key)
        if h is None:
            return
        buf = meshing.build_vertices(h, self.N, self.mpp, self.vexag)
        vdata = GeomVertexData(f"c{key}", GeomVertexFormat.getV3n3c4(), Geom.UHStatic)
        vdata.uncleanSetNumRows(len(buf))
        vdata.modifyArray(0).modifyHandle().copyDataFrom(buf.tobytes())
        geom = Geom(vdata)
        geom.addPrimitive(self.prim)
        node = GeomNode(f"chunk{key}")
        node.addGeom(geom)
        np_ = self.terrain.attachNewNode(node)
        np_.setPos(key[1] * self.N, -key[0] * self.N, 0)
        old = self.meshes.get(key)
        if old:
            old[0].removeNode()
        self.meshes[key] = (np_, self.mesh_ver)

    def _hud_task(self, task):
        if self.hud_on:
            s = self.mgr.stats()
            c = self._chunk_of_pos()
            fps = self.fps_frames * 4
            self.fps_frames = 0
            alt = (self.pos[2] / self.vexag) * self.mpp
            self.hud.setText(
                f"chunk ({c[0]}, {c[1]})   pos row {-self.pos[1]:.0f} col {self.pos[0]:.0f}   alt {alt:.0f} m\n"
                f"speed {self.speed:.0f} px/s   exag x{self.vexag:.2f}   {fps} fps   drawn {len(self.meshes)}\n"
                f"cache {s['cached']}  queued {s['pending']}  generating {s['inflight']}  "
                f"gen {s['generated']} (avg {s['avg_ms']:.0f} ms)  disk {s['disk_hits']}  err {s['errors']}")
        return task.again
