import * as React from "react"
import { ChevronDown, ExternalLink, Languages } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "@/components/ui/dropdown-menu"
import { TooltipProvider } from "@/components/ui/tooltip"
import { News, StatusBadge } from "@/components/site"
import { useLang } from "@/lib/i18n"
import { useJSON, type Progress } from "@/lib/data"

/* where the model is published: the notes' links (Hugging Face, ModelScope), or the single url of older notes */
const modelLinks = (m: NonNullable<Progress["model"]>) => m.links?.length ? m.links : [{ name: "Hugging Face", url: m.url }]
import { when } from "@/lib/format"
import { Now } from "@/sections/Now"
import { Insights } from "@/sections/Insights"
import { Method } from "@/sections/Method"
import { Exam } from "@/sections/Exam"

export function App() {
  const { tr, pick, lang } = useLang()
  const { data: p, error } = useJSON<Progress>("progress.json")
  const latest = p?.timeline?.[0]
  return (
    <TooltipProvider delayDuration={200}>
      <a href="#tasks" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 focus:z-50 focus:rounded-md focus:bg-background focus:px-3 focus:py-2">{tr("Skip to content", "跳到正文")}</a>
      <Header p={p} />
      <main className="mx-auto max-w-[1200px] px-4 sm:px-6">
        <div className="mt-6 flex flex-col gap-2.5 rounded-[10px] border bg-surface px-4 py-3">
          <div className="flex flex-wrap items-center justify-between gap-2.5">
            {p ? <StatusBadge running>{pick(p.status, "pill")}</StatusBadge> : error ? <span className="text-sm text-alarm">{tr("Progress data unavailable. Reload the page.", "进展数据不可用，请刷新页面。")}</span> : <Skeleton className="h-6 w-80" />}
            {p?.model && (
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-sm text-muted-foreground">{tr("Model", "模型")} <span className="num text-foreground">{p.model.repo}</span></span>
                {modelLinks(p.model).map((l) => (
                  <Button key={l.url} asChild variant="outline" size="sm">
                    <a href={l.url} target="_blank" rel="noopener noreferrer">{pick(l, "name")}<ExternalLink /></a>
                  </Button>
                ))}
              </div>
            )}
          </div>
          {latest && <p className="text-sm leading-relaxed text-muted-foreground"><span className="num mr-2 text-foreground">{when(latest.date, lang)}</span><News tone={latest.tone} tr={tr} />{pick(latest, "text")}</p>}
        </div>
        <Now p={p} />
        <Method />
        <Exam p={p} />
        <Insights p={p} />
      </main>
      <footer className="border-t">
        <div className="mx-auto flex max-w-[1200px] flex-wrap items-center justify-between gap-3 px-4 py-6 text-sm text-muted-foreground sm:px-6">
          <span>© 2026 Mercy Zhang · AIOGym · {tr("Process control research in simulation", "过程控制仿真研究")}</span>
          <span className="flex flex-wrap items-center gap-4">
            {p?.model && <span>{tr("Model on ", "模型：")}{modelLinks(p.model).map((l, i) => <React.Fragment key={l.url}>{i > 0 && " · "}<a className="text-foreground underline-offset-4 hover:underline" href={l.url} target="_blank" rel="noopener noreferrer">{pick(l, "name")}</a></React.Fragment>)}</span>}
            <a className="text-foreground underline-offset-4 hover:underline" href="progress.json" download>{tr("Progress data (JSON)", "进展数据（JSON）")}</a>
          </span>
        </div>
      </footer>
    </TooltipProvider>
  )
}

function Header({ p }: { p: Progress | null }) {
  const { tr, pick, lang, setLang } = useLang()
  const nav: [string, string][] = [["#tasks", tr("Now", "现在")], ["#method", tr("Method", "方法")], ["#exam", tr("Scores", "成绩")], ["#insights", tr("Findings", "结论")]]
  return (
    <header className="sticky top-0 z-40 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto flex h-14 max-w-[1200px] items-center gap-4 px-4 sm:px-6">
        <a href="#" className="flex items-center gap-2 font-semibold">
          <img src="icon.svg" alt="" width="24" height="24" className="rounded-[5px]" />
          <span>AIOGym</span><span className="hidden text-muted-foreground sm:inline">/ {tr("Research", "研究进展")}</span>
        </a>
        <nav aria-label={tr("Sections", "章节")} className="ml-2 hidden items-center gap-1 md:flex">
          {nav.map(([h, t]) => <a key={h} href={h} className="rounded-md px-2.5 py-1.5 text-sm text-muted-foreground transition-colors hover:bg-accent hover:text-foreground">{t}</a>)}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          {p && <Badge variant="outline" className="num hidden font-normal text-muted-foreground lg:inline-flex">{tr(`Data ${when(p.generated_at, "en")}`, `数据 ${when(p.generated_at, "zh")}`)}</Badge>}
          {p?.model && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm" aria-label={tr("Where the model is published", "模型下载地址")}>{tr("Model", "模型")}<ChevronDown /></Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                {modelLinks(p.model).map((l) => (
                  <DropdownMenuItem key={l.url} asChild>
                    <a href={l.url} target="_blank" rel="noopener noreferrer">{pick(l, "name")}<ExternalLink className="ml-auto" /></a>
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
          <Button variant="outline" size="sm" onClick={() => setLang(lang === "zh" ? "en" : "zh")} lang={lang === "zh" ? "en" : "zh-CN"} aria-label={lang === "zh" ? "Switch to English" : "切换到中文"}>
            <Languages />{lang === "zh" ? "English" : "中文"}
          </Button>
        </div>
      </div>
      {/* phones: the section links get their own row under the bar */}
      <nav aria-label={tr("Sections", "章节")} className="mx-auto flex h-9 max-w-[1200px] items-center gap-1 border-t px-2 md:hidden">
        {nav.map(([h, t]) => <a key={h} href={h} className="flex-1 rounded-md py-1.5 text-center text-sm text-muted-foreground hover:bg-accent hover:text-foreground">{t}</a>)}
      </nav>
    </header>
  )
}
