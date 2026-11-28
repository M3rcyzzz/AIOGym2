import * as React from "react"
import { Pause, Play } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"
import { useLang } from "@/lib/i18n"
import { Player, reducedMotion, type Snap } from "./player"
import { currentEvent, evKind, evName, type ReplayEvent } from "./events"

export function useSnap(p: Player): Snap {
  return React.useSyncExternalStore(p.subscribe, p.getSnapshot)
}

/* A player for one screen: runs its loop only while the screen is on screen and starts playing the
   first time half of it is visible (not under prefers-reduced-motion). */
export function usePlayer(speed: number, dt: number) {
  const [player] = React.useState(() => new Player(speed, dt))
  const ref = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    const el = ref.current
    if (!el || !("IntersectionObserver" in window)) { player.setVisible(true); return }
    let autoplayed = reducedMotion()
    const io = new IntersectionObserver((es) => {
      const e = es[es.length - 1]
      player.setVisible(e.isIntersecting)
      if (!autoplayed && e.intersectionRatio >= 0.5) { autoplayed = true; player.play() }
    }, { threshold: [0, 0.5] })
    io.observe(el)
    return () => io.disconnect()
  }, [player])
  React.useEffect(() => () => player.dispose(), [player])
  return [player, ref] as const
}

type BarProps = { player: Player; events: ReplayEvent[]; fmt: (s: number) => string; evText: (e: ReplayEvent) => string; speeds: number[]; stepLabel: string }

export function ReplayBar({ player, events, fmt, evText, speeds, stepLabel }: BarProps) {
  const { tr } = useLang()
  const snap = useSnap(player)
  const dur = (player.n - 1) * player.dt
  const t = snap.i * player.dt
  const cur = currentEvent(events, t)
  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
        <Button size="icon" variant="default" className="size-9 rounded-full" onClick={() => player.toggle()}
          aria-label={snap.playing ? tr("Pause replay", "暂停回放") : tr("Play replay", "播放回放")}>
          {snap.playing ? <Pause className="size-4" /> : <Play className="size-4 translate-x-px" />}
        </Button>
        <span className="num text-sm whitespace-nowrap"><b className="font-medium">{fmt(t)}</b><span className="text-muted-foreground"> / {fmt(dur)}</span></span>
        <div className="relative min-w-[200px] flex-1 py-2">
          <Slider value={[snap.i]} min={0} max={Math.max(1, player.n - 1)} step={1} aria-label={stepLabel}
            onValueChange={(v) => { player.pause(); player.setPos(v[0]) }} />
          <div className="pointer-events-none absolute inset-x-0 -top-1.5 h-3">
            {events.map((e, i) => {
              const f = Math.min(Math.max(e.t / Math.max(dur, 1), 0), 1)
              return (
                <button key={i} type="button" onClick={() => player.seekTime(e.t)}
                  className={cn(`ev-${evKind(e)} pointer-events-auto absolute top-0 h-3 w-3 -translate-x-1/2`)}
                  style={{ left: `calc(${(f * 100).toFixed(3)}% + ${((0.5 - f) * 16).toFixed(1)}px)` }}
                  title={`${i + 1} · ${fmt(e.t)} · ${evText(e)}`} aria-label={tr(`Jump to event ${i + 1}`, `跳到事件 ${i + 1}`)}>
                  <span className={cn("mx-auto block h-3 w-0.5 rounded-sm bg-[var(--ev)]", i === cur && "w-1")} />
                </button>
              )
            })}
          </div>
        </div>
        <ToggleGroup type="single" variant="outline" size="sm" value={String(snap.speed)} aria-label={tr("Playback speed", "回放速度")}
          onValueChange={(v) => v && player.setSpeed(Number(v))}>
          {speeds.map((v) => <ToggleGroupItem key={v} value={String(v)} className="num px-2.5 text-[13px]">{v}×</ToggleGroupItem>)}
        </ToggleGroup>
      </div>
      <div className="flex flex-wrap items-center gap-1.5" aria-label={tr("Events", "事件")}>
        {events.length === 0 && <span className="text-sm text-muted-foreground">{tr("No events in this run", "本次回放没有事件")}</span>}
        {events.map((e, i) => (
          <Tooltip key={i}>
            <TooltipTrigger asChild>
              <button type="button" onClick={() => player.seekTime(e.t)}
                className={cn(`ev-${evKind(e)} inline-flex max-w-full items-center gap-1.5 rounded-full border bg-background py-0.5 pr-2.5 pl-0.5 text-[13px] transition-colors hover:border-[var(--ev)]`,
                  i === cur ? "border-[var(--ev)] text-foreground shadow-[inset_0_0_0_1px_var(--ev)]" : i > cur ? "text-muted-foreground" : "text-foreground/80")}>
                <b className="num grid h-[18px] min-w-[18px] place-items-center rounded-full bg-[var(--ev)] px-1 text-xs font-semibold text-white">{i + 1}</b>
                <time className="num text-[13px]">{fmt(e.t)}</time>
                <span className="hidden truncate sm:inline">{evText(e)}</span>
              </button>
            </TooltipTrigger>
            <TooltipContent>{evName(e, tr)} · {evText(e)}</TooltipContent>
          </Tooltip>
        ))}
      </div>
      {events.length > 0 && (
        <p className="text-[13px] leading-snug sm:hidden" aria-live="polite">
          <span className="text-muted-foreground">{cur >= 0 ? tr(`Event ${cur + 1}`, `事件 ${cur + 1}`) : tr("Next: event 1", "下一个：事件 1")} · {fmt(events[Math.max(cur, 0)].t)} · </span>
          {evText(events[Math.max(cur, 0)])}
        </p>
      )}
    </div>
  )
}

export function ReplaySelect({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: { value: string; label: string; group?: string }[] }) {
  return (
    <label className="flex max-w-full min-w-0 items-center gap-2 text-sm text-muted-foreground">
      <span className="whitespace-nowrap">{label}</span>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger size="sm" className="max-w-full min-w-0 text-foreground sm:min-w-[150px]"><SelectValue /></SelectTrigger>
        <SelectContent>
          {options.map((o) => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
        </SelectContent>
      </Select>
    </label>
  )
}

/* One trend under a screen: a static SVG (series, band, events) with a cursor and a live value
   that follow the player; pointer drags seek. */
export type TrendSvg = { svg: string; X: (i: number) => number; pad: { l: number; r: number }; W: number }
export function TrendView({ player, trend, onValue }: { player: Player; trend: TrendSvg; onValue?: () => void }) {
  const host = React.useRef<HTMLDivElement>(null)
  const cb = React.useRef(onValue)
  cb.current = onValue
  React.useEffect(() => {
    const cursor = host.current?.querySelector("[data-cursor]")
    return player.onRender(() => {
      const x = trend.X(player.pos).toFixed(1)
      cursor?.setAttribute("x1", x); cursor?.setAttribute("x2", x)
      cb.current?.()
    })
  }, [player, trend])
  const seek = (e: React.PointerEvent) => {
    const svg = host.current?.querySelector("svg"); if (!svg) return
    const r = svg.getBoundingClientRect(), x = ((e.clientX - r.left) / r.width) * trend.W
    player.setPos(((x - trend.pad.l) / (trend.W - trend.pad.l - trend.pad.r)) * (player.n - 1))
  }
  return (
    <div ref={host} dangerouslySetInnerHTML={{ __html: trend.svg }}
      onPointerDown={(e) => { if (!(e.target as Element).matches("[data-hit]")) return; player.pause(); seek(e); (e.target as Element).setPointerCapture(e.pointerId) }}
      onPointerMove={(e) => { if (e.buttons & 1 && (e.target as Element).matches("[data-hit]")) seek(e) }} />
  )
}
export function TrendCard({ player, title, sub, trend, value }: { player: Player; title: string; sub: React.ReactNode; trend: TrendSvg; value: () => { text: string; warn?: boolean } }) {
  const val = React.useRef<HTMLElement>(null)
  const update = () => { const v = value(); if (val.current) { val.current.textContent = v.text; val.current.classList.toggle("text-warn", !!v.warn) } }
  return (
    <div className="rounded-md border bg-card p-3">
      <div className="mb-1 flex items-start justify-between gap-3">
        <div className="min-w-0"><h4 className="text-sm font-semibold">{title}</h4><p className="text-[13px] text-muted-foreground">{sub}</p></div>
        <strong ref={val} className="num text-sm font-medium whitespace-nowrap">—</strong>
      </div>
      <TrendView player={player} trend={trend} onValue={update} />
    </div>
  )
}

export function SummaryChips({ items }: { items: { text: React.ReactNode; tone?: "ok" | "warn" | "" ; strong?: boolean }[] }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((x, i) => (
        <span key={i} className={cn("rounded-full border px-2.5 py-0.5 text-[13px]",
          x.tone === "warn" ? "border-warn/30 bg-warn-bg text-warn" : x.tone === "ok" ? "border-ok/20 bg-ok-bg text-ok" : "bg-muted text-foreground", x.strong && "font-medium")}>{x.text}</span>
      ))}
    </div>
  )
}
