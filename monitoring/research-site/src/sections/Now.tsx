import { Section, StatusBadge } from "@/components/site"
import { useLang } from "@/lib/i18n"
import type { Progress } from "@/lib/data"
import { TrainBars } from "@/sections/TrainBars"

/* 01 · the task and training as they stand (Mercy 11-12: the old per-plant tabs, one-episode replays and the
   "a new plant every episode" framing no longer mean anything and were removed). Fixed wording lives here;
   numbers and run states come from the notes (sim_rl) and train-status.json, never typed into this file. Key points only. */

export function Now({ p }: { p: Progress | null }) {
  const { tr, pick } = useLang()
  const s = p?.sim_rl
  return (
    <Section id="tasks" index="01" title={tr("What it trains on now", "现在在练什么")}>
      {s && (
        <div className="mb-4 flex flex-col gap-2">
          <StatusBadge running={s.state === "training"}>{pick(s, "pill")}</StatusBadge>
          <p className="max-w-[860px] text-sm leading-relaxed text-foreground/80">{pick(s, "text")}</p>
        </div>
      )}
      <TrainBars p={p} />
    </Section>
  )
}
