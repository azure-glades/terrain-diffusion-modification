"""Panda3D explorer: fly camera + streaming terrain meshes driven by ChunkManager."""
import math
import time

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (AntialiasAttrib, Fog, Geom, GeomEnums, GeomNode, GeomTriangles,
                          GeomVertexData, GeomVertexFormat, KeyboardButton, TextNode,
                          WindowProperties)

import meshing
from chunks import keys_within

SKY = (0.62, 0.76, 0.90)


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
                 "Tab mouse capture | F wireframe | P screenshot | H hide HUD | Esc quit",
            parent=self.a2dBottomLeft, pos=(0.04, 0.09), scale=0.04, align=TextNode.ALeft,
            fg=(1, 1, 1, 1), shadow=(0, 0, 0, 1))

        # ----- initial area (blocking, with progress) -----
        si, sj = start
        self.pos = [float(sj), -float(si), 0.0]
        c0 = self._chunk_of_pos()
        mgr.update(c0)
        ring = keys_within(c0, mgr.pregen_radius)
        t0 = time.time()

        def progress(have, total):
            self.hud.setText(f"Generating terrain around start: {have}/{total} chunks")
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
        self.wire = False
        self._set_capture(True)

        self.taskMgr.add(self._move_task, "move")
        self.taskMgr.add(self._stream_task, "stream")
        self.taskMgr.add(self._build_task, "build")
        self.taskMgr.doMethodLater(0.25, self._hud_task, "hud")
        self.last_tick = 0.0
        self.fps_frames = 0

    # ----------------------------------------------------------- helpers
    def _chunk_of_pos(self):
        i, j = -self.pos[1], self.pos[0]
        return (int(math.floor(i / self.N)), int(math.floor(j / self.N)))

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
