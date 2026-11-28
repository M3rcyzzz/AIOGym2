import * as React from "react"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { useLang } from "@/lib/i18n"
import { cn, useMedia, PHONE } from "@/lib/utils"
import { Minimize2, ZoomIn } from "lucide-react"

/* The SVG screen box. On a phone the whole screen fits the width as an overview (the trend cards below carry the
   readings), with a button to zoom to the legible 760 px drawing that scrolls sideways; wider screens show it as is. */
export function Stage({ stageRef, viewRef, children }: { stageRef: React.RefObject<HTMLDivElement | null>; viewRef: React.RefObject<HTMLDivElement | null>; children: React.ReactNode }) {
  const { tr } = useLang()
  const phone = useMedia(PHONE)
  const [zoom, setZoom] = React.useState(false)
  const fit = phone && !zoom
  return (
    <div className="flex flex-col gap-1">
      <div ref={viewRef} className={cn("rounded-[10px] border bg-card", fit ? "overflow-hidden" : "overflow-x-auto")}>
        <div ref={stageRef} className={cn("relative", fit ? "p-1" : "min-w-[760px] p-2")}>{children}</div>
      </div>
      {phone && (
        <button type="button" onClick={() => setZoom((z) => !z)} aria-pressed={zoom}
          className="inline-flex items-center gap-1 self-end rounded-full px-2 py-1 text-[13px] text-muted-foreground hover:text-foreground">
          {zoom ? <><Minimize2 className="size-3.5" />{tr("Fit to screen", "整屏显示")}</> : <><ZoomIn className="size-3.5" />{tr("Zoom in to read values · swipe sideways", "放大看读数 · 可左右滑动")}</>}
        </button>
      )}
    </div>
  )
}

export function ReplayFallback({ loading }: { loading?: boolean }) {
  const { tr } = useLang()
  if (loading) return <Skeleton className="aspect-[1080/384] w-full rounded-[10px]" />
  return <p className="rounded-[10px] border bg-muted p-6 text-sm text-muted-foreground">{tr("Replay data unavailable. Reload the page to try again.", "回放数据不可用，请刷新页面重试。")}</p>
}

/* site_progress.replay_freshness tags: an out-of-date checkpoint, or one from an earlier run. They wrap: one is a
   whole sentence and must not push a 390 px page sideways. */
export function StaleTags({ items }: { items: ({ text: string; text_zh?: string } | undefined | null)[] }) {
  const { pick } = useLang()
  return <>{items.filter(Boolean).map((x, i) => <Badge key={i} variant="outline" className="max-w-full shrink justify-start rounded-[10px] border-warn/40 bg-warn-bg text-left whitespace-normal text-warn">{pick(x as Record<string, unknown>, "text")}</Badge>)}</>
}

/* Under every replay: one line on what is compared (site rule 2). Generator notes stay off the page (user 11-10: key points only). */
export function ReplayNote({ short }: { short: string; full?: string }) {
  return <p className="text-sm text-muted-foreground">{short}</p>
}
export function Key({ c }: { c: string }) { return <i className="mx-1 inline-block size-2 rounded-[2px] align-middle" style={{ background: c }} /> }
