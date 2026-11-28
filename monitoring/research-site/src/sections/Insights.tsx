import * as React from "react"
import { ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { News, Prov, Section } from "@/components/site"
import { useLang } from "@/lib/i18n"
import type { Progress } from "@/lib/data"
import { when } from "@/lib/format"

const SHOWN = 5

export function Insights({ p }: { p: Progress | null }) {
  const { tr, pick, lang } = useLang()
  const [open, setOpen] = React.useState(false)
  if (!p) return null
  const ins = lang === "zh" ? p.insights_zh : p.insights
  const stale = p.freshness?.blocks?.notes?.stale
  const cards = [["finding", tr("Finding", "发现")], ["bottleneck", tr("Bottleneck", "瓶颈")], ["next", tr("Next", "下一步")]] as const
  const tl = p.timeline || []
  const item = (e: (typeof tl)[number], i: number) => (
    <li key={i} className="grid gap-0.5 border-t py-2.5 text-sm first:border-t-0 sm:grid-cols-[120px_1fr] sm:gap-3">
      <span className="text-muted-foreground">
        <time className="num" dateTime={e.date.replace(" ", "T")}>{when(e.date, lang)}</time>
        {e.auto && <span className="ml-1.5 text-[13px]" title={tr("Read from the training machine", "从训练机自动读取")}>{tr("auto", "自动")}</span>}
      </span>
      <p className="leading-relaxed"><News tone={e.tone} tr={tr} />{pick(e, "text")}</p>
    </li>
  )
  return (
    <Section id="insights" index="04" title={tr("Findings and next steps", "结论与下一步")} aside={<Prov stale={stale ? pick(stale, "text") : undefined}>{tr(`Written ${when(p.notes_as_of, "en")} (UTC+8)`, `写于 ${when(p.notes_as_of, "zh")}（北京时间）`)}</Prov>}>
      <div className="grid gap-3 lg:grid-cols-3">
        {cards.map(([k, label], i) => (
          <article key={k} className={i === 2 ? "rounded-[10px] border border-[var(--fx-black)] bg-[var(--fx-black)] p-5 text-white" : "rounded-[10px] border bg-card p-5"}>
            <span className={i === 2 ? "text-sm font-medium text-[var(--fx-lime)]" : "text-sm font-medium text-muted-foreground"}>{label}</span>
            <h3 className="mt-1 mb-2 text-[17px] leading-snug font-semibold">{ins[`${k}_title`]}</h3>
            <p className={i === 2 ? "text-sm leading-relaxed text-white/80" : "text-sm leading-relaxed text-foreground/80"}>{ins[k]}</p>
          </article>
        ))}
      </div>
      <div className="mt-6 rounded-[10px] border px-4 py-2">
        <h4 className="py-2 text-sm font-semibold">{tr("Progress log", "进展日志")}</h4>
        <Collapsible open={open} onOpenChange={setOpen}>
          <ol>{tl.slice(0, SHOWN).map(item)}</ol>
          <CollapsibleContent><ol className="border-t">{tl.slice(SHOWN).map((e, i) => item(e, i + SHOWN))}</ol></CollapsibleContent>
          {tl.length > SHOWN && (
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="sm" className="my-1 -ml-2 text-muted-foreground">
                {open ? tr("Show fewer", "收起") : tr(`Show all ${tl.length}`, `显示全部 ${tl.length} 条`)}<ChevronDown className={open ? "rotate-180 transition-transform" : "transition-transform"} />
              </Button>
            </CollapsibleTrigger>
          )}
        </Collapsible>
      </div>
    </Section>
  )
}
