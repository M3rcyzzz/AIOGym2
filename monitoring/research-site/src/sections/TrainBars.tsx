import * as React from "react"
import { useLang } from "@/lib/i18n"
import type { Progress } from "@/lib/data"

/* Training progress bars (Mercy 11-15: "在网站上做个训练时间进度条"; only the running runs whose train_labels entry has show). train-status.json is not part of a release: the
   Mac next to the Colab watchers writes it into the live release every 10 minutes (monitoring/train_status.py), so the
   bars move without a new release. Missing file (a fresh deploy before the next push): no card. Run names are codes;
   the notes' train_labels give each a plain name (longest matching prefix, "scr-" ignored), the seed is added. */

export type TrainRun = {
  name: string; job: string; state: "running" | "quiet" | "done" | "stopped"
  rows: number; total?: number; cap_hours?: number; hours?: number; t_row?: number
  eta?: number; row_at?: number; started?: number; ended?: number
}
export type TrainStatus = { updated: number; source: string; runs: TrainRun[] }
export type TrainLabel = { prefix: string; en: string; zh: string; show?: boolean }

const URL_ = "train-status.json"
const STALE = 45 * 60

function useTrainStatus(): TrainStatus | null {
  const [s, set] = React.useState<TrainStatus | null>(null)
  React.useEffect(() => {
    let live = true
    const get = () => fetch(`${URL_}?t=${Date.now()}`, { cache: "no-store" })
      .then((r) => (r.ok ? r.json() : null)).then((d) => { if (live) set(d && Array.isArray(d.runs) ? d : null) })
      .catch(() => { /* the file comes with the next push */ })
    get()
    const id = window.setInterval(get, 60_000)
    return () => { live = false; window.clearInterval(id) }
  }, [])
  return s
}

function useNow(ms = 30_000) {
  const [now, set] = React.useState(() => Date.now() / 1000)
  React.useEffect(() => { const id = window.setInterval(() => set(Date.now() / 1000), ms); return () => window.clearInterval(id) }, [ms])
  return now
}

function labelHit(name: string, labels: TrainLabel[] | undefined) {
  const base = name.replace(/-s\d+$/, "").replace(/^scr-/, "")
  return (labels ?? []).filter((l) => base.startsWith(l.prefix.replace(/^scr-/, "")))
    .sort((a, b) => b.prefix.length - a.prefix.length)[0]
}

export function labelOf(name: string, labels: TrainLabel[] | undefined, tr: (en: string, zh: string) => string) {
  const m = /^(.*?)-s(\d+)$/.exec(name)
  const base = (m ? m[1] : name).replace(/^scr-/, "")
  const hit = (labels ?? []).filter((l) => base.startsWith(l.prefix.replace(/^scr-/, "")))
    .sort((a, b) => b.prefix.length - a.prefix.length)[0]
  const text = hit ? tr(hit.en, hit.zh) : base
  return m ? `${text} · ${tr("seed", "种子")} ${m[2]}` : text
}

function span(sec: number, tr: (en: string, zh: string) => string) {
  const m = Math.max(1, Math.round(sec / 60))
  if (m < 60) return tr(`${m} min`, `${m} 分钟`)
  const h = Math.floor(m / 60), r = m % 60
  return r ? tr(`${h} h ${r} min`, `${h} 小时 ${r} 分`) : tr(`${h} h`, `${h} 小时`)
}

function clock(t: number, lang: string) {
  return new Date(t * 1000).toLocaleString(lang === "zh" ? "zh-CN" : "en-GB", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit", timeZone: "Asia/Shanghai" })
}

export function TrainBars({ p }: { p: Progress | null }) {
  const { tr, lang } = useLang()
  const s = useTrainStatus()
  const now = useNow()
  const labels = p?.train_labels ?? undefined
  // Mercy 11-15: only the main runs while they run; finished, stopped and side runs stay off the page
  const runs = (s?.runs ?? []).filter((r) => (r.state === "running" || r.state === "quiet") && labelHit(r.name, labels)?.show)
  if (!s || runs.length === 0) return null
  const age = now - s.updated
  const running = runs.length
  return (
    <div className="rounded-[10px] border px-4 py-3" aria-live="polite">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
        <h4 className="text-sm font-semibold">{tr("Training progress", "训练进度")}{running > 0 && <span className="ml-2 font-normal text-muted-foreground">{tr(`${running} running`, `${running} 条在跑`)}</span>}</h4>
        <p className="text-[13px] text-muted-foreground">
          {tr(`Colab · updated ${span(age, tr)} ago`, `Colab · ${span(age, tr)}前更新`)}
        </p>
      </div>
      {age > STALE && (
        <p className="mt-1 text-[13px] text-warn">
          {tr("No new rows for a while: the Colab job is between VMs or stopped.",
              "有一阵没有新进度：Colab 任务在换机器或已停。")}
        </p>
      )}
      <ul className="mt-3 flex flex-col gap-3">
        {runs.map((r) => {
          const total = r.total ?? null
          // between pushes the bar moves on by the run's own time per row, never past the plan
          const extra = r.state === "running" && r.t_row && r.row_at ? Math.max(0, (now - r.row_at) / r.t_row) : 0
          const rows = total ? Math.min(total, r.rows + Math.floor(extra)) : r.rows
          const frac = total ? Math.min(1, (r.rows + extra) / total) : null
          const left = r.eta ? r.eta - now : null
          let what: string
          if (r.state === "done") what = tr(`done${r.ended ? " " + clock(r.ended, lang) : ""}`, `已跑完${r.ended ? " " + clock(r.ended, lang) : ""}`)
          else if (r.state === "stopped") what = tr(`stopped at round ${r.rows}`, `停在第 ${r.rows} 轮`)
          else if (r.state === "quiet") what = tr("no new round lately", "近来没有新的一轮")
          else if (left == null) what = r.rows === 0 ? tr("waiting for the first round", "等第一轮") : tr("running", "在跑")
          else what = left > 60 ? tr(`about ${span(left, tr)} left`, `还要约 ${span(left, tr)}`) : tr("finishing", "快跑完了")
          const colour = r.state === "running" ? "var(--chart-3)" : r.state === "done" ? "var(--fx-black-2)" : "var(--warn)"
          return (
            <li key={`${r.job}/${r.name}`}>
              <div className="mb-1 flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
                <span className="min-w-0 font-medium">{labelOf(r.name, labels, tr)}</span>
                <span className="num text-[13px] text-muted-foreground">
                  {total ? tr(`round ${rows} of ${total}`, `第 ${rows} / ${total} 轮`) : tr(`round ${rows}`, `第 ${rows} 轮`)} · {what}
                </span>
              </div>
              <div className="relative h-2.5 rounded-full bg-[var(--surface)]" role="progressbar" aria-valuemin={0} aria-valuemax={total ?? undefined} aria-valuenow={rows}>
                {frac != null && <div className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-700" style={{ width: `${Math.max(1.5, frac * 100)}%`, background: colour, opacity: r.state === "done" ? 0.55 : 1 }} />}
              </div>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
