import * as React from "react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { Progress, Run } from "@/lib/data"
import type { Pick, Tr } from "@/lib/i18n"

export function Section({ id, index, title, aside, children, className }: { id: string; index: string; title: string; aside?: React.ReactNode; children: React.ReactNode; className?: string }) {
  return (
    <section id={id} aria-labelledby={`${id}-h`} className={cn("border-t py-10 first:border-t-0 sm:py-12", className)}>
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <h2 id={`${id}-h`} className="flex items-baseline gap-3 text-2xl font-semibold tracking-tight">
          <span className="num text-base font-medium text-muted-foreground">{index}</span>{title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  )
}

export function Stat({ value, unit, label, tone }: { value: React.ReactNode; unit?: string; label: React.ReactNode; tone?: "good" | "muted" }) {
  return (
    <div className="flex min-w-0 flex-col gap-1 rounded-[10px] border bg-card p-3 sm:p-4">
      <strong className={cn("num text-xl font-medium tracking-tight sm:text-2xl", tone === "muted" && "text-muted-foreground", tone === "good" && "text-ok")}>
        {value}{unit && <span className="ml-1 text-sm font-normal text-muted-foreground">{unit}</span>}
      </strong>
      <span className="text-[13px] leading-snug text-muted-foreground sm:text-sm">{label}</span>
    </div>
  )
}

/* a log entry opens with its verdict (user 11-09: every entry starts with good or bad news, a clear view) */
export function News({ tone, tr }: { tone?: "good" | "bad"; tr: Tr }) {
  if (!tone) return null
  return <span className={cn("font-medium", tone === "good" ? "text-ok" : "text-warn")}>{tone === "good" ? tr("Good news: ", "好消息：") : tr("Bad news: ", "坏消息：")}</span>
}

export function StatusBadge({ running, children }: { running?: boolean; children: React.ReactNode }) {
  return (
    <Badge variant="outline" className={cn("max-w-full gap-1.5 rounded-full px-2.5 py-0.5 text-left text-[13px] font-normal whitespace-normal", running ? "border-[var(--fx-lime)] bg-accent text-foreground" : "text-muted-foreground")}>
      <i className={cn("size-1.5 shrink-0 rounded-full", running ? "bg-[var(--fx-deep-green)]" : "bg-muted-foreground/60")} />{children}
    </Badge>
  )
}

export function StaleBadge({ text }: { text: string }) {
  return <Badge variant="outline" className="max-w-full border-warn/40 bg-warn-bg text-left font-normal whitespace-normal text-warn">{text}</Badge>
}

/* Which run a block comes from and when its data stops (AGENTS.md site rule 1). */
export function Prov({ children, stale }: { children: React.ReactNode; stale?: string }) {
  return <p className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">{children}{stale && <StaleBadge text={stale} />}</p>
}

export function runStatus(r: Run, tr: Tr, pick: Pick): [boolean, string] {
  const seeds = r.seeds || [], running = seeds.filter((s) => s.state === "running").length
  const name = pick(r, "label").split(" · ")[0]
  if (!seeds.length) return [false, pick(r, "status_note") || tr("Not started yet", "尚未开始")]
  if (running) return [true, tr(`${running} of ${seeds.length} seed${seeds.length === 1 ? "" : "s"} running`, `${running}/${seeds.length} 个种子运行中`)]
  if (pick(r, "status_note")) return [false, pick(r, "status_note")]
  if (seeds.length && seeds.every((s) => s.state === "done")) return [false, tr("Finished", "已结束")]
  return [false, tr(`${name} · stopped`, `${name} · 已停止`)]
}

/* the families a suite run trains: the notes' list, else the run's own config (rsac --plant a,b,c, plus TEP with --tep-envs) */
export const suiteFamilies = (r: Run | null | undefined): string[] => {
  if (r?.families?.length) return r.families
  const cfg = r?.seeds?.[0]?.config
  return cfg ? [...(cfg.plant || "").split(",").filter(Boolean), ...(cfg.tep_envs ? ["tep"] : [])] : []
}
/* "Fifteen plants" / "十五类装置" from the number of families a suite run trains */
const EN_N = ["No", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen", "Twenty"]
const ZH_N = ["零", "一", "两", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十"]
export const suiteCount = (r: Run | null | undefined): [string, string] => {
  const n = suiteFamilies(r).length
  return [EN_N[n] || String(n), ZH_N[n] || String(n)]
}

export const currentRun = (p: Progress, task: string) => {
  const all = (p.runs || []).filter((r) => (r.task || (r.id.startsWith("ph-") ? "ph" : r.id.startsWith("mixed-") ? "mixed" : "tank")) === task)
  return all.find((r) => r.role === "current") || all[0] || null
}
