import * as React from "react"
import { Section } from "@/components/site"
import { useLang } from "@/lib/i18n"
import type { Progress, ExamTable, ScoreCurve } from "@/lib/data"
import { ReplayFallback } from "@/replay/shared"
import { cn } from "@/lib/utils"

const SuiteReplay = React.lazy(() => import("@/replay/SuiteReplay"))

/* 03 · exam scores and replays (Mercy 11-13: "把用于验证的任务的结果可视化出来，并把成绩和对比chart出来，像之前那样").
   Scores come from the notes' `tables` (the same batch, references version 3); a table with `chart` is drawn as bars,
   one colour per controller, the numbers printed on the bars, so the chart is its own table view. The replays are
   one episode each of the fixed replay batch (aio.sim.replay, research_site_sim_replay), listed in the notes' `replays`. */

// one colour per controller, by role, the same in every chart; text stays in ink colours
const COLOUR: Record<string, string> = { net: "var(--chart-3)", dmc: "var(--chart-1)", steptest: "var(--chart-2)", pi: "var(--fx-black-2)" }
const num = (s: string) => { const v = Number(s); return Number.isFinite(v) ? v : null }

function Bars({ t }: { t: ExamTable }) {
  const { tr, pick } = useLang()
  return (
    <figure className="rounded-[10px] border p-4">
      <figcaption>
        <h4 className="text-sm font-semibold">{pick(t, "title")}</h4>
        {pick(t, "note") && <p className="mt-0.5 text-[13px] text-muted-foreground">{pick(t, "note")}</p>}
      </figcaption>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted-foreground" aria-label={tr("Legend", "图例")}>
        {t.rows.map((r) => <li key={r.key ?? r.name} className="flex items-center gap-1.5"><i className="inline-block size-2.5 rounded-[2px]" style={{ background: COLOUR[r.key ?? ""] ?? "var(--fx-black-2)" }} />{pick(r, "name")}</li>)}
      </ul>
      <div className={cn("mt-3 grid gap-4", t.cols.length === 2 && "md:grid-cols-2", t.cols.length >= 3 && "md:grid-cols-3")}>
        {t.cols.map((c, j) => (
          <div key={j}>
            <p className="mb-1.5 text-[13px] font-medium">{tr(c[0], c[1])}</p>
            <div className="flex flex-col gap-1.5">
              {t.rows.map((r) => {
                const v = num(r.v[j])
                return (
                  <div key={r.key ?? r.name} className="flex items-center gap-2" title={`${pick(r, "name")}: ${r.v[j]}`}>
                    <div className="relative h-5 flex-1 rounded-[3px] bg-[var(--surface)]">
                      {v != null && <div className="absolute inset-y-0 left-0 rounded-[3px]" style={{ width: `${Math.max(0, Math.min(100, v))}%`, background: COLOUR[r.key ?? ""] ?? "var(--fx-black-2)" }} />}
                    </div>
                    <span className="num w-10 shrink-0 text-right text-sm">{r.v[j]}</span>
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </figure>
  )
}

/* one plant a row (Mercy 11-14: "换个更直观的图，绿色换成模型分数"): the network a green dot, the DMC a blue tick, the
   self-tuning PID a grey tick, the line from the DMC to the network; the axis runs to 110 so a score above 100 shows,
   with the oracle MPC (100) dashed. Rows by the network's score. */
const AX = 110
function Dots({ t }: { t: ExamTable }) {
  const { tr, pick } = useLang()
  const row = (k: string) => t.rows.find((r) => r.key === k)
  const net = row("net"), dmc = row("dmc"), pi = row("pi")
  if (!net) return null
  const x = (v: number | null) => `${(Math.max(0, Math.min(AX, v ?? 0)) / AX) * 100}%`
  const items = t.cols.map((c, j) => ({ c, n: num(net.v[j]), d: dmc ? num(dmc.v[j]) : null, p: pi ? num(pi.v[j]) : null }))
    .sort((a, b) => (b.n ?? -1e9) - (a.n ?? -1e9))
  const Tick = ({ v, c, h }: { v: number | null; c: string; h: string }) => v == null ? null
    : <span className={cn("absolute top-1/2 w-[3px] -translate-x-1/2 -translate-y-1/2 rounded-full", h)} style={{ left: x(v), background: c }} />
  return (
    <figure className="rounded-[10px] border p-4">
      <figcaption>
        <h4 className="text-sm font-semibold">{pick(t, "title")}</h4>
        {pick(t, "note") && <p className="mt-0.5 text-[13px] text-muted-foreground">{pick(t, "note")}</p>}
      </figcaption>
      <ul className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-[13px] text-muted-foreground" aria-label={tr("Legend", "图例")}>
        <li className="flex items-center gap-1.5"><i className="inline-block size-3 rounded-full ring-2 ring-white" style={{ background: COLOUR.net }} />{pick(net, "name")}</li>
        {dmc && <li className="flex items-center gap-1.5"><i className="inline-block h-4 w-[3px] rounded-full" style={{ background: COLOUR.dmc }} />{pick(dmc, "name")}</li>}
        {pi && <li className="flex items-center gap-1.5"><i className="inline-block h-3 w-[3px] rounded-full" style={{ background: COLOUR.pi }} />{pick(pi, "name")}</li>}
        <li className="flex items-center gap-1.5"><i className="inline-block h-3.5 border-l border-dashed border-[var(--fx-black-4)]" />{tr("100: oracle MPC", "100：oracle MPC")}</li>
      </ul>
      <div className="mt-3 flex flex-col">
        {items.map(({ c, n, d, p }, i) => {
          return (
            <div key={i} className="grid grid-cols-[1fr_auto] items-center gap-x-3 border-t py-1.5 first:border-t-0 sm:grid-cols-[11rem_1fr_2.5rem]"
              title={`${tr(c[0], c[1])}: ${pick(net, "name")} ${n ?? "—"}${d != null ? ` · ${pick(dmc!, "name")} ${d}` : ""}${p != null ? ` · ${pick(pi!, "name")} ${p}` : ""}`}>
              <span className="col-start-1 row-start-1 truncate text-sm sm:col-start-auto sm:row-start-auto">{tr(c[0], c[1])}</span>
              <span className="relative col-span-2 row-start-2 h-6 sm:col-span-1 sm:row-start-auto">
                <span className="absolute inset-x-0 top-1/2 h-px bg-[var(--fx-black-1)]" />
                <span className="absolute inset-y-0 border-l border-dashed border-[var(--fx-black-4)]" style={{ left: x(100) }} />
                {n != null && d != null && <span className="absolute top-1/2 h-[2px] -translate-y-1/2" style={{ left: x(Math.min(n, d)), width: `${(Math.abs(n - d) / AX) * 100}%`, background: COLOUR.net, opacity: .45 }} />}
                <Tick v={p} c={COLOUR.pi} h="h-3" />
                {n != null && <span className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-white" style={{ left: x(n), background: COLOUR.net }} />}
                <Tick v={d} c={COLOUR.dmc} h="h-5" />
              </span>
              <span className="num col-start-2 row-start-1 text-right text-sm sm:col-start-auto sm:row-start-auto">
                <b className="font-semibold">{n ?? "—"}</b>
              </span>
            </div>
          )
        })}
      </div>
      <div className="relative mt-1 h-4 text-[12px] text-muted-foreground sm:ml-[calc(11rem+0.75rem)] sm:mr-[calc(2.5rem+0.75rem)]" aria-hidden>
        {[0, 50, 100].map((v) => <span key={v} className="num absolute -translate-x-1/2" style={{ left: x(v) }}>{v}</span>)}
      </div>
    </figure>
  )
}

/* validation score by training round (Mercy 11-25: charts of the run in training): one line per series of the notes'
   `curve`, the whole batch in ink and thicker, 0 (valves left alone) dashed; the last value of each series in the legend. */
const LINE: Record<string, { c: string; w: number; dash?: string }> = {
  all: { c: "var(--fx-black)", w: 2.4 }, other: { c: "var(--chart-3)", w: 1.8 }, level: { c: "var(--chart-1)", w: 1.8 },
  s24: { c: "var(--chart-2)", w: 1.8 }, s48: { c: "var(--chart-2)", w: 1.8, dash: "5 4" },
  blind: { c: "var(--fx-black-2)", w: 1.8 },
}
function Curve({ c }: { c: ScoreCurve }) {
  const { tr, pick } = useLang()
  const box = React.useRef<HTMLDivElement>(null)
  const [W, setW] = React.useState(720)                       // drawn at the width it has, so the axis text keeps its size on a phone
  React.useEffect(() => {
    const el = box.current
    if (!el) return
    const ro = new ResizeObserver(() => setW(Math.max(280, Math.round(el.clientWidth))))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])
  const H = Math.round(Math.max(240, Math.min(340, W * .46))), L = 34, R = 16, T = 22, B = 26
  const xs = c.points.map((p) => p[0] as number)
  const refs = c.refs || []
  const vals = [...c.points.flatMap((p) => p.slice(1)).filter((v): v is number => v != null), ...refs.map((r) => r.v)]
  if (!xs.length || !vals.length) return null
  const x0 = Math.min(...xs), x1 = Math.max(...xs)
  const st = Math.max(...vals) - Math.min(...vals) > 60 ? 20 : 10
  const lo = Math.floor(Math.min(0, ...vals) / st) * st, hi = Math.ceil(Math.max(10, ...vals) / st) * st
  const X = (v: number) => L + ((v - x0) / Math.max(1, x1 - x0)) * (W - L - R)
  const Y = (v: number) => T + ((hi - v) / (hi - lo)) * (H - T - B)
  const ticks: number[] = []
  for (let v = lo; v <= hi; v += st) ticks.push(v)
  const xt = [0, .25, .5, .75, 1].map((f) => Math.round((x0 + f * (x1 - x0)) / 5) * 5)
  const last = (j: number) => { for (let i = c.points.length - 1; i >= 0; i--) { const v = c.points[i][j + 1]; if (v != null) return v } return null }
  const fmt = (v: number | null) => v == null ? "—" : (v < 0 ? "−" : "+") + Math.abs(v).toFixed(1)
  return (
    <figure className="rounded-[10px] border p-4">
      <figcaption>
        <h4 className="text-sm font-semibold">{pick(c, "title")}</h4>
        {pick(c, "note") && <p className="mt-0.5 text-[13px] text-muted-foreground">{pick(c, "note")}</p>}
      </figcaption>
      <ul className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-[13px] text-muted-foreground" aria-label={tr("Legend", "图例")}>
        {c.series.map((s, j) => {
          const l = LINE[s.key] ?? { c: "var(--fx-black-2)", w: 1.8 }
          return (
            <li key={s.key} className="flex items-center gap-1.5">
              <svg width="18" height="8" aria-hidden><line x1="0" y1="4" x2="18" y2="4" stroke={l.c} strokeWidth={l.w} strokeDasharray={l.dash} /></svg>
              {pick(s, "name")} <b className="num font-semibold text-foreground">{fmt(last(j))}</b>
            </li>
          )
        })}
      </ul>
      <div ref={box} className="mt-2 w-full">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className="block" role="img" aria-label={pick(c, "title")}>
        {ticks.map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke={v === 0 ? "var(--fx-black-4)" : "var(--fx-black-1)"} strokeWidth={1} strokeDasharray={v === 0 ? "4 3" : undefined} opacity={v === 0 ? 1 : .6} />
            <text x={L - 6} y={Y(v) + 4} textAnchor="end" fontSize="12" fill="var(--fx-black-4)" className="num">{v}</text>
          </g>
        ))}
        {refs.map((r, i) => {
          const near = refs.some((o, j) => j !== i && o.v > r.v && Y(r.v) - Y(o.v) < 18)      // a line close under another: its name below it
          return (
            <g key={r.name}>
              <line x1={L} x2={W - R} y1={Y(r.v)} y2={Y(r.v)} stroke="var(--fx-black-4)" strokeWidth={1.2} strokeDasharray="2 4" />
              <text x={W - R - 2} y={Y(r.v) + (near ? 15 : -5)} textAnchor="end" fontSize="13" fill="var(--fx-black-4)">{pick(r, "name")} {r.v}</text>
            </g>
          )
        })}
        {xt.map((v, i) => <text key={i} x={X(v)} y={H - 8} textAnchor={i === 0 ? "start" : i === xt.length - 1 ? "end" : "middle"} fontSize="12" fill="var(--fx-black-4)" className="num">{v}</text>)}
        {c.series.map((s, j) => {
          const l = LINE[s.key] ?? { c: "var(--fx-black-2)", w: 1.8 }
          const pts = c.points.filter((p) => p[j + 1] != null).map((p) => `${X(p[0] as number).toFixed(1)},${Y(p[j + 1] as number).toFixed(1)}`).join(" ")
          return <polyline key={s.key} points={pts} fill="none" stroke={l.c} strokeWidth={l.w} strokeDasharray={l.dash} strokeLinejoin="round" strokeLinecap="round" />
        }).reverse()}
      </svg>
      </div>
      <p className="mt-1 text-center text-[13px] text-muted-foreground">{tr("Training round", "训练轮数")}</p>
    </figure>
  )
}

function Table({ t }: { t: ExamTable }) {
  const { tr, pick } = useLang()
  return (
    <div className="rounded-[10px] border px-4 py-3">
      <h4 className="text-sm font-semibold">{pick(t, "title")}</h4>
      <p className="mt-0.5 mb-2 text-[13px] text-muted-foreground">{pick(t, "note")}</p>
      <table className="w-full text-sm">
        <thead><tr className="text-left text-[13px] text-muted-foreground">
          <th className="py-1.5 pr-2 font-medium">{t.rowhead ? pick(t, "rowhead") : tr("Controller", "控制器")}</th>
          {t.cols.map((c, j) => <th key={j} className="num py-1.5 px-1 text-right font-medium">{tr(c[0], c[1])}</th>)}
        </tr></thead>
        <tbody>{t.rows.map((r, i) => (
          <tr key={i} className="border-t align-top">
            <td className="py-1.5 pr-2">{pick(r, "name")}</td>
            {r.v.map((x, j) => <td key={j} className="num py-1.5 px-1 text-right whitespace-nowrap">{x}</td>)}
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

export function Exam({ p }: { p: Progress | null }) {
  const { tr, pick } = useLang()
  const tables = p?.tables || []
  const reps = p?.replays || []
  const [k, setK] = React.useState(0)
  if (!p) return null
  const rep = reps[Math.min(k, reps.length - 1)]
  return (
    <Section id="exam" index="03" title={tr("Exam scores", "考试成绩")}>
      <p className="mb-3 max-w-[860px] text-sm leading-relaxed text-foreground/80">
        {tr("Plants the model never trained on. Valves left alone = 0, the oracle MPC = 100.", "都是模型没练过的装置。阀门不动 0 分，oracle MPC 100 分。")}
      </p>
      {p.exam_note && <p className="mb-3 max-w-[860px] rounded-[10px] border border-[var(--warn)] px-3 py-2 text-sm font-medium">{tr(p.exam_note.en, p.exam_note.zh)}</p>}
      <div className="flex flex-col gap-4">
        {p.curve && <Curve c={p.curve} />}
        {tables.filter((t) => t.chart).map((t, i) => t.dots ? <Dots key={i} t={t} /> : <Bars key={i} t={t} />)}
        {tables.filter((t) => !t.chart).map((t, i) => <Table key={i} t={t} />)}
      </div>
      {rep && (
        <div className="mt-8">
          <h3 className="mb-3 text-[17px] font-semibold">{tr("Replays: one plant, every controller", "回合回放：同一台装置，各控制器各跑一遍")}</h3>
          {reps.length > 1 && (
            <div role="tablist" aria-label={tr("Episode", "回合")} className="mb-3 flex gap-1 overflow-x-auto">
              {reps.map((r, i) => (
                <button key={r.file} role="tab" aria-selected={i === k} onClick={() => setK(i)}
                  className={cn("shrink-0 rounded-md border px-3 py-1.5 text-sm", i === k ? "border-[var(--fx-black)] bg-[var(--fx-black)] text-white" : "text-muted-foreground hover:bg-accent")}>
                  {pick(r, "label")}
                </button>
              ))}
            </div>
          )}
          <React.Suspense fallback={<ReplayFallback loading />}>
            <SuiteReplay key={rep.file} plant="sim" file={rep.file} />
          </React.Suspense>
        </div>
      )}
    </Section>
  )
}
