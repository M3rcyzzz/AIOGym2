import type { Tr } from "@/lib/i18n"

/* Replay events, shared by the four-tank, TEP and pH screens. An event is {t: seconds from the start,
   kind: 'setpoint' | 'process' | 'disturbance' | 'other', label, label_zh, short, short_zh, ...}; a run can
   hold any number. Each is marked on the scrub track, as a numbered line on every trend and as a chip
   under the bar; clicking one jumps to it. Colour says the kind, not good or bad. */
/* params: a process change's new conditions; state: what any event sets from then on (the simulation core: zone limits, priorities) */
export type ReplayEvent = { t: number; kind: string; label?: string; label_zh?: string; short?: string; short_zh?: string; idv?: number; params?: Record<string, number>; state?: Record<string, number> }

const KIND: Record<string, [string, string]> = {
  setpoint: ["Setpoint change", "设定值切换"], process: ["Process change", "工况变化"], disturbance: ["Disturbance", "扰动"], other: ["Event", "事件"],
}
export const evKind = (e: ReplayEvent) => (e.kind in KIND ? e.kind : "other")
export const evName = (e: ReplayEvent, tr: Tr) => { const k = KIND[evKind(e)]; return tr(k[0], k[1]) }
export const cleanEvents = (evs: unknown): ReplayEvent[] =>
  (Array.isArray(evs) ? evs : []).filter((e) => e && Number.isFinite(+e.t)).map((e) => ({ ...e, t: +e.t })).sort((a, b) => a.t - b.t)
export function currentEvent(evs: ReplayEvent[], t: number) { let k = -1; for (let i = 0; i < evs.length; i++) if (evs[i].t <= t + 1e-6) k = i; return k }

/* SVG: one dashed line per event across a trend, numbered at the top. */
export function eventLines(evs: ReplayEvent[], x: (t: number) => number, top: number, bottom: number) {
  let s = "", lastX = -99, row = 0
  evs.forEach((e, i) => {
    const px = x(e.t), k = evKind(e)
    s += `<path class="hmi-t-event ev-${k}" d="M${px.toFixed(1)} ${top}V${bottom}"/>`
    row = px - lastX < 14 ? (row + 1) % 3 : 0
    lastX = px
    s += `<text class="hmi-t-evn ev-${k}" x="${(px + 2.5).toFixed(1)}" y="${top + 7 + row * 9}">${i + 1}</text>`
  })
  return s
}
