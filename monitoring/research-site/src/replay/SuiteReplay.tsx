import * as React from "react"
import { useLang, type Tr } from "@/lib/i18n"
import { useJSON } from "@/lib/data"
import { fx, hhmm, number, short } from "@/lib/format"
import { cleanEvents, currentEvent, eventLines, type ReplayEvent } from "./events"
import { ReplayBar, ReplaySelect, SummaryChips, TrendCard, usePlayer, type TrendSvg } from "./controls"
import { reducedMotion } from "./player"
import { Key, ReplayFallback, ReplayNote, Stage, StaleTags } from "./shared"
import { SCREENS, condLines, type LitModel, type NetFlow, type SimPlant, type Snap } from "./suiteScreens"
import { ZOO } from "./zooScreens"

/* Replays of the APC suite plants (quadruple tank, Shell fractionator, BSM1): <plant>-replay.json from
   research_site_suite_replay. Laid out like the four-tank, TEP and pH replays: the process screen
   (suiteScreens.ts) driven by the replay, the bar and events, the controller choice, outside-time chips,
   one trend per tracked or constrained output and per monitored output the file marks to show, the MVs
   grouped by unit, and the footnote. One P0 test scenario; every controller on the same plant and events. */
/* zone, tag, lims: the simulation core (process 'sim'), whose zone limits move with events: lims holds the steps [t, lower, upper] */
type Ch = { key: string; name?: string; label: string; label_zh?: string; unit: string; tracked?: boolean; band?: number | null; lower?: number | null; upper?: number | null; dec: number; show?: boolean; lo?: number; hi?: number
  zone?: boolean; tag?: string; lims?: [number, number | null, number | null][] }
/* ref: a run's own reachable reference, when it differs from the file's (sim: dead times that follow a throughput MV);
   ok: sim, per tracked CV, the stretch [lower, upper] that does not count as outside where it is wider than the band
   around the reference (a constrained operating point: from the target to the reachable optimum, a band beyond) */
type RawRun = { controller: string; kind: string; label: string; label_zh?: string; short?: string; short_zh?: string; note?: string; note_zh?: string; n: number; cols: Record<string, number[]>; total_s: number; track_s: number; viol_s: number; failed?: boolean; stale?: { text: string }; previous?: { text: string }; ref?: Record<string, number[]>; ok?: Record<string, [number[], number[]]>; [k: string]: unknown }
type Run = RawRun & { cum: number[] }
type Data = { version: number; process: string; scenario: string; pick?: string; dt: number; horizon_s: number; score_from: number; cvs: Ch[]; mvs: Ch[]; ref: Record<string, number[]>; start?: Record<string, number>; events: ReplayEvent[]; runs: RawRun[]; label?: string; label_zh?: string; footnote?: string; footnote_zh?: string; net?: NetFlow; lit?: LitModel; sim?: SimPlant; plant_label?: string; plant_label_zh?: string }
const refOf = (r: RawRun, data: Data, key: string) => r.ref?.[key] || data.ref[key]
/* outside a tracked CV's band around the reference, or outside its wider stretch where the run carries one */
const offRef = (r: RawRun, data: Data, ch: Ch, i: number, y: number) => {
  const ok = r.ok?.[ch.key]
  return ok ? y < ok[0][i] || y > ok[1][i] : Math.abs(y - refOf(r, data, ch.key)[i]) > (ch.band || 0)
}
/* an MV's operator limits as steps (sim: they move with events), else its fixed lower and upper */
const mvSteps = (m: Ch): [number, number | null, number | null][] | undefined =>
  m.lims?.length ? m.lims : m.lower != null || m.upper != null ? [[0, m.lower ?? null, m.upper ?? null]] : undefined
/* a channel's limits at time t: the step in force, else its fixed limits */
const limAt = (c: Ch, t: number): [number | null | undefined, number | null | undefined] => {
  if (!c.lims?.length) return [c.lower, c.upper]
  let k = 0; for (let i = 0; i < c.lims.length; i++) if (c.lims[i][0] <= t + 1e-6) k = i
  return [c.lims[k][1], c.lims[k][2]]
}

const MV_COLOURS = ["#4f7fae", "#7a6cbf", "#c0705f"]
const DAY = 86400
/* held-out families (docs/heldout.md): never trained, references PI and no action only; the literature columns (lit) also
   show the formula PI and the step-test DMC, on the scenario with the median formula-PI score */
const HELD_OUT = ["boiler", "wo", "mma", "cryst"]

function clock(h: number, tr: Tr) {
  if (h <= 2 * DAY) return hhmm
  return (s: number) => { const d = Math.floor(s / DAY); return tr(`day ${d} ${hhmm(s - d * DAY)}`, `第 ${d} 天 ${hhmm(s - d * DAY)}`) }
}
/* time ticks: minutes up to 2 h, hours up to 2 days, days beyond */
function ticks(h: number, tr: Tr): [number, string][] {
  if (h <= 7200) return [0, 30, 60, 90, 120].filter((m) => m * 60 <= h).map((m) => [m * 60, m * 60 === h ? `${m}${tr(" min", " 分钟")}` : String(m)])
  if (h <= 2 * DAY) { const step = h <= DAY ? 6 : 12; const out: [number, string][] = []; for (let x = 0; x * 3600 <= h; x += step) out.push([x * 3600, (x + step) * 3600 > h ? `${x} h` : String(x)]); return out }
  const out: [number, string][] = []; for (let d = 0; d * DAY <= h; d++) out.push([d * DAY, (d + 1) * DAY > h ? tr(`${d} d`, `${d} 天`) : String(d)]); return out
}
const dur = (s: number, h: number, tr: Tr) => (h <= 2 * 3600 ? tr(`${(s / 60).toFixed(1)} min`, `${(s / 60).toFixed(1)} 分钟`) : tr(`${(s / 3600).toFixed(1)} h`, `${(s / 3600).toFixed(1)} 小时`))
const unit = (c: Ch) => (c.unit ? ` ${c.unit}` : "")

/* Outside seconds so far: the file's per-row count of outputs the score counts as outside (cols.out), or,
   for a fragment written before it existed, the same rule recounted from the stored traces. */
function prepare(r: RawRun, data: Data): Run {
  const c = r.cols, k0 = Math.round(data.score_from / data.dt), cum = [0]
  for (let i = 1; i < r.n; i++) {
    let bad = 0
    if (c.out) bad = c.out[i]
    else if (i - 1 >= k0) for (const ch of data.cvs) {
      const y = c[ch.key]?.[i]; if (y == null) continue
      if (ch.tracked && offRef(r, data, ch, i, y)) bad++
      const [lo, hi] = limAt(ch, (i - 1) * data.dt)
      if ((lo != null && y < lo) || (hi != null && y > hi)) bad++
    }
    cum[i] = cum[i - 1] + bad * data.dt
  }
  return { ...r, cum }
}
function sample(r: Run, p: number, data: Data, evs: ReplayEvent[]): Snap & { i: number } {
  const i = Math.min(Math.floor(p), r.n - 1), j = Math.min(i + 1, r.n - 1), w = p - i, c = r.cols
  const v: Record<string, number> = {}, ref: Record<string, number> = {}
  for (const ch of data.cvs) { const a = c[ch.key]; if (a) v[ch.name || ch.key] = a[i] * (1 - w) + a[j] * w; if (ch.tracked) ref[ch.name || ch.key] = refOf(r, data, ch.key)[j] }
  for (const m of data.mvs) { const a = c[m.key]; if (a) v[m.key] = a[i] * (1 - w) + a[j] * w }
  const t = p * data.dt, cond: Record<string, number> = { ...(data.start || {}) }
  for (const e of evs) if (e.t <= t + 1e-6) { if (e.kind === "process" && e.params) Object.assign(cond, e.params); if (e.state) Object.assign(cond, e.state) }
  for (const ch of data.cvs) { const ok = r.ok?.[ch.key]; if (ok) { cond[`oklo_${ch.name || ch.key}`] = ok[0][j]; cond[`okhi_${ch.name || ch.key}`] = ok[1][j] } }
  return { i, t, v, ref, cond }
}

type Steps = [number, number | null, number | null][]
/* ok: the stretch drawn as the band instead of ref ± band; steps: limits that move, one set per channel (stepCls: its colour) */
type Opts = { ref?: number[]; band?: number | null; ok?: [number[], number[]]; limits?: (number | null | undefined)[]; steps?: Steps[]; stepCls?: (string | undefined)[]; compare?: Run | null; cls?: string[]; dec: number }
function spark(r: Run, data: Data, evs: ReplayEvent[], keys: string[], tr: Tr, o: Opts): TrendSvg {
  const W = 380, H = 140, p = { l: 40, r: 8, t: 8, b: 16 }, n = r.n, c = r.cols
  const vals = keys.flatMap((k) => c[k]).concat(o.compare ? keys.flatMap((k) => o.compare!.cols[k]) : [])
  let lo = Math.min(...vals), hi = Math.max(...vals)
  if (o.ref) for (const v of o.ref) { lo = Math.min(lo, v - (o.band || 0)); hi = Math.max(hi, v + (o.band || 0)) }
  if (o.ok) { lo = Math.min(lo, ...o.ok[0]); hi = Math.max(hi, ...o.ok[1]) }
  const g0 = hi - lo || Math.abs(hi) || 1
  const near = (v: number | null | undefined): v is number => v != null && v >= lo - 0.5 * g0 && v <= hi + 0.5 * g0
  const sets = (o.steps || []).filter((x) => x.length)
  const lims = (sets.length ? [] : o.limits || []).filter(near)
  for (const v of [...lims, ...sets.flat().flatMap((s) => [s[1], s[2]]).filter(near)]) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
  const g = hi - lo || 1; lo -= g * 0.06; hi += g * 0.06
  const X = (i: number) => p.l + (i / Math.max(1, n - 1)) * (W - p.l - p.r), Y = (v: number) => H - p.b - ((v - lo) / (hi - lo)) * (H - p.t - p.b)
  const line = (arr: number[]) => arr.map((v, i) => `${i ? "L" : "M"}${X(i).toFixed(1)} ${Y(v).toFixed(1)}`).join("")
  let body = ""
  if (o.ref) {
    const b = o.band || 0, ok = o.ok
    const up = o.ref.map((v, i) => [X(i), Y(ok ? ok[1][i] : v + b)]), dn = o.ref.map((v, i) => [X(i), Y(ok ? ok[0][i] : v - b)]).reverse()
    body += `<path class="hmi-t-band" d="M${up.concat(dn).map((q) => q.map((x) => x.toFixed(1)).join(" ")).join("L")}Z"/><path class="hmi-t-sp" d="${line(o.ref)}"/>`
  }
  for (const v of lims) body += `<line class="hmi-t-limit" x1="${p.l}" x2="${W - p.r}" y1="${Y(v).toFixed(1)}" y2="${Y(v).toFixed(1)}"/>`
  /* limits that move with events: one step line per side, drawn where the limit is near the curves */
  sets.forEach((st, q) => {
    let d = ""
    for (const side of [1, 2] as const) st.forEach((s, k) => {
      const v = s[side], x0 = X(s[0] / data.dt), x1 = X(k + 1 < st.length ? st[k + 1][0] / data.dt : n - 1)
      if (near(v)) d += `M${x0.toFixed(1)} ${Y(v).toFixed(1)}H${x1.toFixed(1)}`
    })
    if (d) body += `<path class="hmi-t-limit" ${o.stepCls?.[q] ? `style="stroke:${o.stepCls[q]}" ` : ""}d="${d}"/>`
  })
  body += eventLines(evs, (t) => X(t / data.dt), p.t, H - p.b)
  const span = hi - lo, d = Math.max(0, Math.min(4, Math.ceil(-Math.log10(span / 4))))
  for (let k = 0; k <= 2; k++) { const v = lo + span * (0.1 + 0.4 * k); body += `<text class="hmi-t-axis" x="${p.l - 5}" y="${(Y(v) + 3).toFixed(1)}" text-anchor="end">${Math.abs(v) >= 1e4 ? short(v) : v.toFixed(d)}</text>` }
  for (const [s, label] of ticks(data.horizon_s, tr)) body += `<text class="hmi-t-axis" x="${X(s / data.dt).toFixed(1)}" y="${H - 3}" text-anchor="${s ? (s + 1e-6 >= data.horizon_s ? "end" : "middle") : "start"}">${label}</text>`
  if (o.compare) for (const k of keys) body += `<path class="hmi-t-ghost" d="${line(o.compare.cols[k])}"/>`
  keys.forEach((k, i) => { body += `<path class="hmi-t-line" ${o.cls?.[i] ? `style="stroke:${o.cls[i]}"` : ""} d="${line(c[k])}"/>` })
  body += `<line class="hmi-t-cursor" data-cursor y1="${p.t}" y2="${H - p.b}" x1="${p.l}" x2="${p.l}"/><rect class="hmi-t-hit" data-hit x="${p.l}" y="0" width="${W - p.l - p.r}" height="${H}"/>`
  return { svg: `<svg viewBox="0 0 ${W} ${H}" class="hmi-trend-svg" aria-hidden="true">${body}</svg>`, X, pad: p, W }
}

export default function SuiteReplay({ plant, file }: { plant: string; file?: string }) {
  const { data, error } = useJSON<Data>(file || `${plant}-replay.json`)
  if (error || (data && (data.version !== 1 || data.process !== plant || !data.runs?.length))) return <ReplayFallback />
  if (!data) return <ReplayFallback loading />
  return <SuiteView key={file || plant} data={data} />
}

function SuiteView({ data }: { data: Data }) {
  const { tr, pick } = useLang()
  const sc = SCREENS[data.process] || ZOO[data.process]
  const evs = React.useMemo(() => cleanEvents(data.events), [data])
  const runs = React.useMemo(() => data.runs.map((r) => prepare(r, data)), [data])
  const ctl = (id: string) => runs.find((r) => r.controller === id)
  const [runId, setRunId] = React.useState((ctl("rl") || runs[0]).controller)
  const run = ctl(runId) || runs[0]
  const [cmpId, setCmpId] = React.useState(run.controller === "dmc" ? "" : ctl("dmc") ? "dmc" : "")
  const [all, setAll] = React.useState(false)
  const compare = cmpId && cmpId !== run.controller ? ctl(cmpId) || null : null
  const base = Math.max(1, Math.round(data.horizon_s / 60))
  const speeds = [base / 2, base, base * 2, base * 5].map((v) => Math.max(1, Math.round(v)))
  const [player, viewRef] = usePlayer(speeds[1], data.dt)
  React.useEffect(() => { player.setN(run.n) }, [player, run])
  const fmt = clock(data.horizon_s, tr)
  const h = data.horizon_s
  const svg = React.useMemo(() => sc.svg(tr, data), [sc, tr, data])
  const host = React.useRef<HTMLDivElement>(null), stage = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    const root = host.current?.querySelector("svg"); if (!root) return
    const q = (sel: string) => root.querySelector(sel) as SVGElement | null
    let cur: Snap | null = null
    const dash: Record<string, number> = {}, angle: Record<string, number> = {}, bub: Record<string, number> = {}
    const offR = player.onRender(() => {
      const s = sample(run, player.pos, data, evs); cur = s
      sc.update(q, s, tr, data)
      const [c1, c2] = sc.conds ? sc.conds(s.cond, tr) : condLines(sc, s.cond, data.start || {}, tr)
      const set = (sel: string, t: string) => { const e = q(sel); if (e) e.textContent = t }
      set("[data-conds]", c1); set("[data-conds2]", c2); set("[data-clock]", fmt(s.t))
      const ci = currentEvent(evs, s.t), N = evs.length, sh = (e: ReplayEvent) => pick(e, "short") || pick(e, "label")
      set("[data-phase]", !N ? tr("No events", "没有事件") : ci < 0 ? tr(`Next event ${fmt(evs[0].t)} · ${sh(evs[0])}`, `下一事件 ${fmt(evs[0].t)} · ${sh(evs[0])}`)
        : tr(`${ci + 1}/${N} · ${sh(evs[ci])} since ${fmt(evs[ci].t)}`, `${ci + 1}/${N} · ${sh(evs[ci])}（自 ${fmt(evs[ci].t)} 起）`))
      set("[data-score]", s.t > data.score_from ? tr(`Outside since ${fmt(data.score_from)}: ${dur(run.cum[s.i], h, tr)}`, `${fmt(data.score_from)} 起带外 ${dur(run.cum[s.i], h, tr)}`)
        : tr(`Counting starts at ${fmt(data.score_from)}`, `带外统计从 ${fmt(data.score_from)} 开始`))
    })
    const offF = reducedMotion() ? () => {} : player.onFrame((dt) => {
      if (!cur) return
      for (const [id, f] of Object.entries(sc.flows(cur, data))) {
        const e = q(`[data-pipe="${id}"]`); if (!e) continue
        dash[id] = ((dash[id] || 0) - dt * f * 45) % 28; e.style.strokeDashoffset = dash[id].toFixed(2); e.style.opacity = f > 0.02 ? "" : "0"
      }
      for (const [id, f] of Object.entries(sc.rotors?.(cur) || {})) {
        const e = q(`[data-rotor="${id}"]`), c = e?.querySelector("circle"); if (!e || !c) continue
        angle[id] = ((angle[id] || 0) + dt * f * 360) % 360; e.setAttribute("transform", `rotate(${angle[id].toFixed(1)} ${c.getAttribute("cx")} ${c.getAttribute("cy")})`)
      }
      for (const [r, f] of Object.entries(sc.bubbles?.(cur) || {})) {
        bub[r] = ((bub[r] || 0) + dt * (0.2 + 1.2 * f)) % 1
        for (let k = 0; k < 6; k++) {
          const e = q(`[data-bub="${r}-${k}"]`); if (!e) continue
          const fr = (bub[r] + k / 6 + (k % 2) * 0.37) % 1
          e.setAttribute("cy", (248 - fr * 20).toFixed(1)); e.style.opacity = f > 0.02 ? ((1 - fr) * 0.9).toFixed(2) : "0"
        }
      }
    })
    return () => { offR(); offF() }
  }, [svg, run, player, data, evs, tr, pick, sc]) // eslint-disable-line react-hooks/exhaustive-deps

  const gray = compare ? tr(" · gray: ", " · 灰色：") + pick(compare, "short") : ""
  const cvs = data.cvs.filter((c) => c.show !== false && run.cols[c.key])
  const mvColour = (m: Ch) => MV_COLOURS[data.mvs.indexOf(m) % MV_COLOURS.length]
  const groups = React.useMemo(() => {
    const g: Ch[][] = []
    for (const m of data.mvs) { const last = g[g.length - 1]; if (last && last[0].unit === m.unit && last.length < MV_COLOURS.length) last.push(m); else g.push([m]) }
    return g
  }, [data])
  const trends = React.useMemo(() => ({
    cv: cvs.map((c) => spark(run, data, evs, [c.key], tr, { compare, dec: c.dec, ref: c.tracked ? refOf(run, data, c.key) : undefined, band: c.band, ok: c.tracked ? run.ok?.[c.key] : undefined,
      limits: [c.lower, c.upper], steps: c.lims ? [c.lims] : undefined })),
    /* an MV keeps its colour (its place among the MVs, as on the process screen) in whichever card it is drawn; its
       operator limits (sim) are drawn in that colour where the MV comes near them */
    mv: groups.map((g) => spark(run, data, evs, g.map((m) => m.key), tr, { compare, dec: g[0].dec, cls: g.length > 1 ? g.map(mvColour) : undefined, limits: [],
      steps: g.map(mvSteps).filter((x): x is Steps => !!x), stepCls: g.length > 1 ? g.filter(mvSteps).map(mvColour) : undefined })),
  }), [run, compare, data, evs, tr, groups]) // eslint-disable-line react-hooks/exhaustive-deps
  const at = (k: string, arr?: number[]) => { const a = arr || run.cols[k]; return a[Math.min(Math.round(player.pos), a.length - 1)] }
  /* a learner trained before 11-07 (PR #158) is shown the packet it was trained on: no gain signs, and the evaporator's old pairing */
  const legacy = (run.source as { packet?: string } | undefined)?.packet === "legacy"
    ? (data.process === "evap" ? tr("It sees the packet it was trained on: no gain signs, the old pairing.", "学习器看的是训练时的旧数据包：没有增益符号，配对也是旧的。")
      : tr("It sees the packet it was trained on, without gain signs.", "学习器看的是训练时的旧数据包，没有增益符号。")) : ""
  const outside = (c: Ch) => (c.tracked ? (run[`${c.name}_s`] as number) || 0 : (run[`${c.name}_viol_s`] as number) || 0)
  return (
    <div className="flex flex-col gap-4">
      <Stage stageRef={stage} viewRef={viewRef}><div ref={host} dangerouslySetInnerHTML={{ __html: svg }} /></Stage>
      <ReplayBar player={player} events={evs} fmt={fmt} evText={(e) => pick(e, "label")} speeds={speeds} stepLabel={tr(`Replay position, ${data.dt}-second steps`, `回放位置，每步 ${data.dt} 秒`)} />
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
        <ReplaySelect label={tr("Controller", "控制器")} value={run.controller} onChange={(v) => { setRunId(v); if (v === cmpId) setCmpId("") }} options={runs.map((r) => ({ value: r.controller, label: pick(r, "label") }))} />
        <ReplaySelect label={tr("Compare", "对比")} value={compare?.controller || "none"} onChange={(v) => setCmpId(v === "none" ? "" : v)}
          options={[{ value: "none", label: tr("None", "无") }, ...runs.filter((r) => r !== run).map((r) => ({ value: r.controller, label: pick(r, "label") }))]} />
        <StaleTags items={[run.stale, run.previous]} />
      </div>
      {(pick(run, "note") || legacy) && <p className="text-sm text-muted-foreground">{[pick(run, "note"), legacy].filter(Boolean).join(tr(". ", "。"))}</p>}
      <SummaryChips items={[
        { text: tr(`Outside from ${fmt(data.score_from)}`, `${fmt(data.score_from)} 起带外`), strong: true },
        { text: `${tr("Total", "合计")} ${dur(run.total_s, h, tr)}`, tone: run.total_s > 0 ? "warn" : "ok", strong: true },
        { text: `${tr("off target", "偏离目标")} ${dur(run.track_s, h, tr)}`, tone: run.track_s > 0 ? "warn" : "ok" },
        { text: `${tr("outside limits", "超出限值")} ${dur(run.viol_s, h, tr)}`, tone: run.viol_s > 0 ? "warn" : "ok" },
        ...(run.failed ? [{ text: tr("Failed; the rest counts as outside", "失效，之后全部计为带外"), tone: "warn" as const }] : []),
        ...(compare ? [{ text: `${pick(compare, "short")} ${dur(compare.total_s, h, tr)}` }] : []),
      ]} />
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {cvs.map((c, i) => {
          if (!all && !c.tracked) return null   // Mercy 11-15 "网站弄简洁点，特别是图": the tracked CVs first, the rest on request
          /* a random plant's band is a drawn float: three significant digits */
          const bw = c.band != null ? +c.band.toPrecision(3) : c.band
          /* and its limits are drawn floats too (net: 359.8139953613281): five significant digits */
          const lo = c.lower != null ? +c.lower.toPrecision(5) : c.lower, hi = c.upper != null ? +c.upper.toPrecision(5) : c.upper
          const what = c.tracked ? tr(`band ±${bw}${unit(c)} around the reference`, `参考值 ±${bw}${unit(c)}`) + (run.ok?.[c.key] ? tr(", wider at a constrained point", "，约束工况放宽") : "")
            : (c.lims?.length || 0) > 1 ? tr(`limits move with events (${c.lims!.length - 1}×)`, `限值随事件移动 ${c.lims!.length - 1} 次`)
            : lo != null && hi != null ? tr(`limits ${lo} to ${hi}${unit(c)}`, `限值 ${lo} 至 ${hi}${unit(c)}`)
            : lo != null ? tr(`at least ${lo}${unit(c)}`, `不低于 ${lo}${unit(c)}`)
            : hi != null ? tr(`at most ${hi}${unit(c)}`, `不高于 ${hi}${unit(c)}`) : tr("monitored, not scored", "只监视，不计分")
          const scored = c.tracked || c.zone || c.lower != null || c.upper != null
          return (
            <TrendCard key={c.key} player={player} title={pick(c, "label")} trend={trends.cv[i]}
              sub={<>{c.unit ? `${c.unit} · ` : ""}{what}{scored && tr(` · outside ${dur(outside(c), h, tr)}`, ` · 带外 ${dur(outside(c), h, tr)}`)}{gray}</>}
              value={() => {
                const v = at(c.key), [lo, hi] = limAt(c, player.pos * data.dt)
                const warn = c.tracked ? offRef(run, data, c, Math.min(Math.round(player.pos), run.n - 1), v) : (lo != null && v < lo) || (hi != null && v > hi)
                return { text: `${fx(v, Math.min(c.dec, 3))}${unit(c)}`, warn }
              }} />
          )
        })}
        {all && groups.map((g, i) => (
          <TrendCard key={g[0].key} player={player} title={g.length > 1 ? tr("Manipulated variables", "操纵变量") : pick(g[0], "label")} trend={trends.mv[i]}
            sub={<>{g[0].unit ? `${g[0].unit} · ` : ""}{g.length > 1 ? g.map((m) => <React.Fragment key={m.key}><Key c={mvColour(m)} />{pick(m, "label")} </React.Fragment>) : tr(`range ${number(g[0].lo!)}–${number(g[0].hi!)}`, `范围 ${number(g[0].lo!)}–${number(g[0].hi!)}`)}{g.some((m) => (m.lims?.length || 0) > 1) ? tr(" · limits move with events", " · 限值随事件移动") : ""}{gray}</>}
            value={() => ({ text: g.map((m) => fx(at(m.key), Math.min(m.dec, 2))).join(" / ") })} />
        ))}
      </div>
      {!all && <button type="button" onClick={() => setAll(true)} className="mt-2 rounded-md border px-3 py-1.5 text-sm text-muted-foreground hover:bg-accent">{tr("Show limits and valves too", "也显示限值量和阀门")}</button>}
      {/* held-out and random-plant families have no DMC: their scenario is the one with the median PI score (research_site_suite_replay) */}
      <ReplayNote short={data.process === "sim" ? (data.pick?.includes("constrained")
          ? tr("Same episode, plant and events for every controller: the constrained episode with the median static-model MPC score.", "所有控制器同一回合、同一装置和事件：取到过约束工况、静态模型MPC 成绩居中的回合。")
          : tr("Same episode, plant and events for every controller: the one with the median static-model MPC score.", "所有控制器同一回合、同一装置和事件：取静态模型MPC 成绩居中的回合。"))
        : data.process === "lit" ? tr("Every controller runs the same test scenario and events: the one with the median self-tuning PID score.", "所有控制器跑同一个测试场景和事件：取自整定PID 成绩居中的那个。")
        : data.process === "net" ? tr("Every controller runs the same test scenario and events: one whose self-tuning PID score is near the test-set median.", "所有控制器跑同一个测试场景和事件：自整定PID 成绩接近测试集中位数的那个。")
        : HELD_OUT.includes(data.process) || data.process === "proc2" || data.process === "mimo" ? tr("Every controller runs the same test scenario and events: the one with the median self-tuning PID score.", "所有控制器跑同一个测试场景和事件：取自整定PID 成绩居中的那个。")
        : data.process === "proc" ? tr("Every controller runs the same test scenario and events: the one with the median self-tuning PID score.", "所有控制器跑同一个测试场景和事件：取自整定PID 成绩居中的那个。")
        : tr("Every controller runs the same test scenario and events: the one with the median static-model MPC score.", "所有控制器跑同一个测试场景和事件：取静态模型MPC 成绩居中的那个。")} full={pick(data, "footnote") || undefined} />
    </div>
  )
}
