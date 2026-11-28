/* Playback clock shared by a process screen, its controls and its trends.
   pos is a fractional sample index; React reads a coarse snapshot (rounded index, playing, speed)
   through useSyncExternalStore, while the screen and trend cursors redraw imperatively every frame.
   The animation loop runs only while the screen is on screen (setVisible) and stops when unmounted. */
export type Snap = { i: number; playing: boolean; speed: number }

export class Player {
  pos = 0
  playing = false
  speed: number
  n = 1
  readonly dt: number
  private snap: Snap
  private subs = new Set<() => void>()
  private renders = new Set<() => void>()
  private frames = new Set<(dt: number) => void>()
  private raf = 0
  private last = 0
  private dirty = true
  private visible = false
  private alive = true

  constructor(speed: number, dt: number) {
    this.speed = speed
    this.dt = dt
    this.snap = { i: 0, playing: false, speed }
  }
  subscribe = (cb: () => void) => { this.subs.add(cb); return () => { this.subs.delete(cb) } }
  getSnapshot = () => this.snap
  private emit() {
    const i = Math.round(this.pos)
    if (i !== this.snap.i || this.playing !== this.snap.playing || this.speed !== this.snap.speed) {
      this.snap = { i, playing: this.playing, speed: this.speed }
      for (const cb of this.subs) cb()
    }
  }
  setN(n: number) { this.n = Math.max(1, n); this.setPos(this.pos) }
  setPos(p: number) { this.pos = Math.max(0, Math.min(this.n - 1, p)); this.dirty = true; this.emit(); if (!this.raf) this.flush() }
  seekTime(t: number) { this.pause(); this.setPos(t / this.dt) }
  play() { if (this.pos >= this.n - 1) this.pos = 0; this.playing = true; this.emit(); this.kick() }
  pause() { this.playing = false; this.emit() }
  toggle() { if (this.playing) this.pause(); else this.play() }
  setSpeed(v: number) { this.speed = v; this.emit() }
  onRender(cb: () => void) { this.renders.add(cb); this.dirty = true; this.kick(); return () => { this.renders.delete(cb) } }
  onFrame(cb: (dt: number) => void) { this.frames.add(cb); return () => { this.frames.delete(cb) } }
  invalidate() { this.dirty = true; if (!this.raf) this.flush() }
  setVisible(v: boolean) { this.visible = v; if (v) { this.dirty = true; this.kick() } }
  dispose() { this.alive = false; cancelAnimationFrame(this.raf); this.raf = 0; this.subs.clear(); this.renders.clear(); this.frames.clear() }
  private flush() { if (!this.dirty) return; this.dirty = false; for (const cb of this.renders) cb() }
  private kick() { if (!this.raf && this.alive && this.visible) { this.last = 0; this.raf = requestAnimationFrame(this.tick) } }
  private tick = (now: number) => {
    this.raf = 0
    if (!this.alive || !this.visible) return
    const dt = Math.min(0.1, (now - (this.last || now)) / 1000)
    this.last = now
    if (this.playing) {
      this.pos += (dt * this.speed) / this.dt
      if (this.pos >= this.n - 1) { this.pos = this.n - 1; this.playing = false }
      this.dirty = true
      this.emit()
    }
    this.flush()
    for (const cb of this.frames) cb(dt)
    this.raf = requestAnimationFrame(this.tick)
  }
}

export const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches
