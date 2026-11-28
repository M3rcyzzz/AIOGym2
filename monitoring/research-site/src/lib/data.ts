import * as React from "react"

/* progress.json, written at publish time by experiments/shared_rl_view/site_progress.py. */
export type Text = { text: string; text_zh?: string }
export type Ref = { id: string; label: string; label_zh?: string; iqm: number | null; kind?: string; tone?: "worse"; note?: string; note_zh?: string; demand?: number; disturbed?: number; source?: string }
export type Seed = {
  seed: string; state: "running" | "done" | "stopped"; samples?: number; hours?: number; sps?: number; updated?: string; started?: string
  best?: { samples: number; val_iqm: number } | null
  val?: number[][]; train?: number[][]; tep_val?: (number | null)[][]; train_tep?: number[][]
  best_mixed?: { samples: number; tep_val_iqm: number; score: number; val_iqm: number | null }
  /* several plant families in one run (rsac --plant a,b,c): validation IQM per family, the selection score
     (geometric mean of the families' IQMs) and the checkpoint with the lowest score */
  fam_val?: Record<string, number[][]>; score_val?: number[][]
  best_suite?: { samples: number; score: number; iqm: Record<string, number | null>; guard_ok?: boolean | null; guard_violation?: number | null } | null
  /* forgetting guard (rsac --guard-families): per validation, the IQM of the per-scenario difference to the
     reference policy for each guarded family, and whether all stayed within guard_frac of the reference IQM */
  guard?: [number, Record<string, number>, boolean][]
  /* a run on the unified simulation core (aio.train): per validation [samples, x_pi, cost_x_pi, win share, failed], x_pi the
     multiple of the formula PI on the same batch (1 = formula PI); the literature columns' multiples recorded with it (not
     used for selection); the student's decisions outside on its training episodes; best.pt (lowest x_pi) and the last one */
  sim_val?: (number | null)[][]; lit_val?: Record<string, number[][]>; student?: number[][]
  best_sim?: SimPoint | null; last_sim?: SimPoint | null
  config?: {
    max_hours?: number; envs?: number; tep_envs?: number; tep_frac?: number; reward_scale?: number[]; plant?: string
    family_scale?: Record<string, number>; scale_anchor?: string | null; snapshot_hours?: number
    anchor_families?: string[] | string | null; anchor_coef?: number | null; guard_families?: string | null; guard_frac?: number | null; guard_ref_iqm?: Record<string, number> | null
    anchor_of?: Record<string, string> | null  // each family's teacher: "init" (the starting actor) or <run>/<seed>/<checkpoint>
    plant_envs?: Record<string, number> | null; train_horizon?: string | null  // rsac --plant-weights (plants per family), --train-horizon fam=steps
    qnorm?: number | null; anchor_filter?: boolean | null  // rsac --qnorm (actor loss per family), --anchor-filter (Q-filtered BC)
    critic_family?: boolean | null  // rsac --critic-family: the critics see a family embedding (training only; the actor is unchanged)
    /* aio.train: its budget, batches, validation batch, starting checkpoint and the references measured on that batch */
    trainer?: string | null; hours?: number | null; window?: number | null; seqs?: number | null; sim_envs?: number | null; val_n?: number | null; val_seed?: number | null
    lit?: number | null; init?: string | null; beta_iters?: number | null
    val_refs?: Record<string, { score: number; cost: number; x_pi: number; con?: number | null }> | null; lit_formula_pi?: Record<string, number> | null
    /* the points scale (11-11): 0 = no action (val_refs.hold), 100 = scale.best (the per-episode best causal controller; an
       Oracle among them when scale.oracle) */
    scale?: { best?: number | null; oracle?: boolean | null; pi?: number | null; hold?: number | null } | null
  } | null
}
export type SimPoint = { it?: number | null; samples: number; hours?: number | null; x_pi: number; score?: number | null; cost_x_pi?: number | null; win?: number | null; failed?: number | null; lit?: Record<string, number> | null; con?: number | null; con_pi?: number | null }
export type FamRefs = { refs: Ref[]; snapshot?: { iqm: number; note: string; note_zh?: string } | null; scope?: string; scope_zh?: string }
export type Run = {
  id: string; role?: string; task?: string; label: string; label_zh?: string; short?: string; short_zh?: string; note?: string; note_zh?: string
  status_note?: string; status_note_zh?: string; oracle_status?: string; oracle_status_zh?: string; oracle_note?: string; oracle_note_zh?: string
  val_refs?: Ref[]; tank_val_refs?: Ref[]; refs?: Ref[]; init_val?: { iqm: number } | null
  /* the run this one starts from; a suite run takes its start per family from that run's best checkpoint */
  init_run?: string; start_note?: string; start_note_zh?: string; start_skip?: string[]
  families?: string[]; fam_refs?: Record<string, FamRefs>
  seeds: Seed[]; stopped?: boolean
}
/* tone "worse": a learner that got worse on this task (forgetting), drawn in red */
export type P0Ref = { id: string; label: string; label_zh?: string; iqm: number; ci95?: number[]; kind?: string; tone?: "worse"; verdict?: "pass" | "fail"; note?: string; note_zh?: string; groups?: Record<string, number> }
/* metric: the value column's heading when it is not an IQM in seconds (the simulation core: decisions outside per episode) */
export type P0Table = { title: string; title_zh?: string; scope: string; scope_zh?: string; refs: P0Ref[]; note?: string; note_zh?: string; groups?: { id: string; label: string; label_zh?: string }[]; metric?: string; metric_zh?: string }
export type Notice = { state: string; title: string; title_zh?: string; text: string; text_zh?: string; pill: string; pill_zh?: string; replay?: string; replay_zh?: string }
export type Progress = {
  generated_at: string; generated_label: string; notes_as_of: string; training_through?: string
  status: { pill: string; pill_zh?: string }
  insights: Record<string, string>; insights_zh: Record<string, string>
  timeline: { date: string; text: string; text_zh?: string; auto?: boolean; tone?: "good" | "bad" }[]   // auto: read from the training machine; tone: opens with good or bad news
  runs: Run[]
  p0_refs?: Record<string, P0Table>
  tep_rl?: Notice | null
  suite_rl?: Notice | null
  zoo_rl?: Notice | null
  expand_rl?: Notice | null
  heldout_rl?: Notice | null
  /* the literature columns (lit, docs/heldout.md section 3): held out, their own notice until a learner's zero-shot run joins the replay */
  lit_rl?: Notice | null
  /* APC-scale random plants (mimo), added for round 2: their own notice until a run trains them */
  mimo_rl?: Notice | null
  /* the unified simulation core (aio/sim): its own notice until its first round has a trained learner in the replay */
  sim_rl?: Notice | null
  tables?: ExamTable[] | null
  /* validation score by training round of the run in training (notes `curve`; points: [round, score per series...], null where not read) */
  curve?: ScoreCurve | null
  /* plain names for the training runs' codes in train-status.json (TrainBars) */
  train_labels?: { prefix: string; en: string; zh: string; show?: boolean }[] | null
  exam_note?: { en: string; zh: string } | null
  replays?: { file: string; label: string; label_zh?: string }[] | null
  model?: { url: string; repo: string; label: string; label_zh?: string; links?: { name: string; name_zh?: string; url: string }[] } | null  // the published model (site_progress_notes.json)
  superseded?: Record<string, { title: string; title_zh?: string; text: string; text_zh?: string; tag?: string; tag_zh?: string }> | null
  freshness?: { blocks?: Record<string, { stale?: Text; through?: string; run?: string }> }
}

const cache = new Map<string, Promise<unknown>>()
export function loadJSON<T>(url: string): Promise<T> {
  if (!cache.has(url)) {
    cache.set(url, fetch(url, { cache: "no-cache" }).then((r) => {
      if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`)
      return r.json()
    }))
  }
  return cache.get(url) as Promise<T>
}

/* Load a data file once a component mounts (tabs mount their panel only when opened). */
export function useJSON<T>(url: string, enabled = true): { data: T | null; error: Error | null } {
  const [state, setState] = React.useState<{ data: T | null; error: Error | null }>({ data: null, error: null })
  React.useEffect(() => {
    if (!enabled) return
    let live = true
    loadJSON<T>(url).then((data) => live && setState({ data, error: null }), (error) => {
      if (live) setState({ data: null, error })
      console.warn(`${url} unavailable`, error)
    })
    return () => { live = false }
  }, [url, enabled])
  return state
}

/* True once the element has come within `margin` of the viewport; used to defer charts and polling. */
export function useNearViewport<E extends Element>(margin = "300px"): [React.RefObject<E | null>, boolean] {
  const ref = React.useRef<E>(null)
  const [near, setNear] = React.useState(false)
  React.useEffect(() => {
    const el = ref.current
    if (!el || near) return
    if (!("IntersectionObserver" in window)) { setNear(true); return }
    const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { setNear(true); io.disconnect() } }, { rootMargin: margin })
    io.observe(el)
    return () => io.disconnect()
  }, [near, margin])
  return [ref, near]
}

export const runName = (r: Run, pick: (o: Record<string, unknown>, k: string) => string) => pick(r, "label").split(" · ")[0]

/* The points scale of a run on the simulation core (11-11): 0 = no action, 100 = the per-episode best causal controller
   (scale.best; without it, the best reference on the batch), on the same validation batch. Formula PI is only a line on it.
   oracle false: no Oracle among the references, so 100 is easier to reach. */
export function simScale(cfg: Seed["config"]) {
  const refs = cfg?.val_refs || {}, hold = refs.hold?.score, pi = refs.pi?.score
  const scores = Object.values(refs).map((v) => v.score).filter((x) => Number.isFinite(x))
  const top = cfg?.scale?.best ?? (scores.length ? Math.min(...scores) : undefined)
  const ok = hold != null && top != null && hold > top
  const pts = (score?: number | null) => (ok && score != null && Number.isFinite(score) ? (100 * (hold! - score)) / (hold! - top!) : null)
  return { ok, oracle: !!cfg?.scale?.oracle && cfg?.scale?.best != null, pts, ofX: (x?: number | null) => (x != null && pi ? pts(x * pi) : null) }
}

/* a score table of the notes (same batch); `chart` draws it as bars, a row's `key` picks its controller colour */
export type ScoreCurve = { title: string; title_zh?: string; note: string; note_zh?: string; series: { key: string; name: string; name_zh?: string }[]; points: (number | null)[][]; refs?: { name: string; name_zh?: string; v: number }[] }
export type ExamTable = { chart?: boolean; dots?: boolean; rowhead?: string; rowhead_zh?: string; title: string; title_zh?: string; note: string; note_zh?: string; cols: [string, string][]; rows: { key?: string; name: string; name_zh?: string; v: string[] }[] }
