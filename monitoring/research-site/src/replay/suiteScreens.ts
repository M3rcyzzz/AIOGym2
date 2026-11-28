import type { Tr } from "@/lib/i18n"
import { esc, fx, number } from "@/lib/format"

/* Process screens of the APC suite plants (quadruple tank, Shell fractionator, BSM1), drawn in the same
   vocabulary as the four-tank, TEP and pH screens (hmi-* classes): vessels with their contents, pipes whose
   dashes move with the flow, mono readouts, indicators with the band around the reference, MV bars, and
   the clock, event and outside-time lines that SuiteReplay fills. Topology and tags follow the environment
   code: qtank_batch.py, shell_batch.py, bsm_batch.py. Each screen reads one snapshot per frame:
   v (CVs by name, MVs by key), ref (tracked CVs by name) and cond (process conditions in force). */
export type Snap = { t: number; v: Record<string, number>; ref: Record<string, number>; cond: Record<string, number> }
export type Q = (sel: string) => SVGElement | null
/* the replay's channels, for screens whose layout follows the file (the random plants) */
export type Chan = { key: string; name?: string; label: string; label_zh?: string; tracked?: boolean; band?: number | null; lower?: number | null; upper?: number | null; dec: number; integrating?: boolean; lo?: number; hi?: number; paired?: string; gamma?: number; range?: number[]
  /* the simulation core (process 'sim'): a zone row, its short tag (CV1, Z1, D1), its limit steps [t, lower, upper] when events move them, an MV's acting sign */
  zone?: boolean; tag?: string; lims?: [number, number | null, number | null][]; sign?: number }
/* a random flowsheet (netplant_batch.py, research_site_suite_replay.net_structure): units in chain order, recycle loops
   [splitter, destination], side-feed unit (−1: none), purging splitters, the handle each MV moves and what each CV measures
   (unit −1: the feed or the whole plant) */
export type NetFlow = {
  units: { kind: "mixer" | "cstr" | "hx" | "flash" | "splitter" | "pipe"; tag: string }[]; loops: [number, number][]; side: number; purge: number[]
  mvs: { key: string; h: "F0" | "Fs" | "Tu" | "rho"; unit: number }[]; cvs: { name: string; y: "T" | "A" | "P" | "Y" | "Qs" | "cA0" | "T0"; unit: number }[]
}
/* a literature column (lit_batch.py): the model the scenario is drawn from (wb, ory, ty, al) and its number of loops */
export type LitModel = { model: string; n: number }
/* a plant of the simulation core (aio/sim, research_site_sim_replay): its coupling structure and how many unmeasured loads it has */
export type SimPlant = { arch: string; arch_label: string; arch_label_zh?: string; nq: number }
export type Layout = { cvs: Chan[]; mvs: Chan[]; start?: Record<string, number>; net?: NetFlow; lit?: LitModel; sim?: SimPlant; plant_label?: string; plant_label_zh?: string }
type Pair = [string, string]
export type Cond = Record<string, [Pair, (v: number) => string]>
export type Screen = {
  svg: (tr: Tr, d: Layout) => string
  update: (q: Q, s: Snap, tr: Tr, d: Layout) => void
  /* flow speeds 0-1 per data-pipe id and rotor speeds, for the animation */
  flows: (s: Snap, d: Layout) => Record<string, number>
  rotors?: (s: Snap) => Record<string, number>
  bubbles?: (s: Snap) => Record<string, number>
  /* conditions named in the conditions line when they differ from the start, or a fixed readout */
  cond: Cond
  conds?: (cond: Record<string, number>, tr: Tr) => [string, string]
}

export const pct = (v: number) => `${Math.round(100 * v)}%`
export const sgn = (v: number, d = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(d)}`
export const arrowDef = (id: string) => `<marker id="${id}" markerUnits="userSpaceOnUse" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto"><path d="m0 0 9 4.5-9 4.5z" fill="#8b9096"/></marker>`
export const pipe = (id: string, d: string, cls: string, arrow?: string, base = "hmi-pipe") =>
  `<path class="${base}" d="${d}"${arrow ? ` marker-end="url(#${arrow})"` : ""}/><path class="hmi-flow ${cls}" data-pipe="${id}" d="${d}"/>`
export const clockBlock = (x: number, y: number, anchor = "start") =>
  `<text class="hmi-clock" data-clock x="${x}" y="${y}" text-anchor="${anchor}">00:00</text>
  <text class="hmi-sub" data-phase x="${x}" y="${y + 20}" text-anchor="${anchor}"></text><text class="hmi-sub" data-score x="${x}" y="${y + 38}" text-anchor="${anchor}"></text>`

/* An indicator: tag and value on one line, a rail whose middle is the reference with the band shaded,
   scaled to ±5 bands so the band is a fifth of the rail; the pointer stops at the ends. */
export function indicator(id: string, x: number, y: number, w: number, tag: string) {
  return `<g data-ind="${id}"><text class="hmi-tag" x="${x}" y="${y}">${tag}</text><text class="hmi-value" data-value="${id}" x="${x + w}" y="${y}" text-anchor="end">—</text>
  <line class="hmi-rail" x1="${x}" x2="${x + w}" y1="${y + 14}" y2="${y + 14}"/><rect class="hmi-band" data-band="${id}" x="${x + w * 0.4}" y="${y + 9}" width="${w * 0.2}" height="10" rx="2"/>
  <line class="hmi-sp" x1="${x + w / 2}" x2="${x + w / 2}" y1="${y + 6}" y2="${y + 22}"/>
  <path class="hmi-pointer" data-pointer="${id}" d="M0 -2L-5 -10H5Z" transform="translate(${x + w / 2} ${y + 14})"/>
  <text class="hmi-sub" data-sub="${id}" x="${x}" y="${y + 36}"></text></g>`
}
export function setIndicator(q: Q, id: string, x: number, w: number, y: number, v: number, ref: number, band: number, text: string, sub: string) {
  const out = Math.abs(v - ref) > band, px = Math.min(Math.max(x + w / 2 + ((v - ref) / (5 * band)) * (w / 2), x), x + w)
  const val = q(`[data-value="${id}"]`), ptr = q(`[data-pointer="${id}"]`)
  if (val) { val.textContent = text; val.classList.toggle("warn", out) }
  if (ptr) { ptr.setAttribute("transform", `translate(${px.toFixed(1)} ${y + 14})`); ptr.classList.toggle("warn", out) }
  const s = q(`[data-sub="${id}"]`); if (s) s.textContent = sub
}
export const bar = (id: string, x: number, y: number, w = 80) => `<rect class="hmi-bar-bg" x="${x}" y="${y}" width="${w}" height="4" rx="2"/><rect class="hmi-bar" data-bar="${id}" x="${x}" y="${y}" width="0" height="4" rx="2"/>`
export function setBar(q: Q, id: string, f: number, w = 80) { q(`[data-bar="${id}"]`)?.setAttribute("width", (w * Math.min(Math.max(f, 0), 1)).toFixed(1)) }
export function setText(q: Q, sel: string, text: string, warn?: boolean) { const e = q(sel); if (e) { e.textContent = text; if (warn != null) e.classList.toggle("warn", warn) } }

/* ---------- Johansson quadruple tank: same layout as the four-tank screen ---------- */
const QT = { top: 20, V: { T3: [322, 74, 110, 98], T4: [648, 74, 110, 98], T1: [317, 200, 120, 90], T2: [643, 200, 120, 90] } as Record<string, number[]>,
  lim: { T1: [2, 19], T2: [2, 19], T3: [0.2, 19], T4: [0.2, 19] } as Record<string, number[]>, h: { T1: "h1", T2: "h2", T3: "h3", T4: "h4" } as Record<string, string> }
const qtY = (id: string, h: number) => { const [, y, , H] = QT.V[id]; return y + H - 2 - (Math.min(Math.max(h, 0), QT.top) / QT.top) * (H - 4) }
const QT_PIPES: [string, string, string, boolean][] = [
  ["S1", "M160 340V322", "hmi-p1", false], ["S2", "M920 340V322", "hmi-p2", false],
  ["P1M", "M182 300H230V245", "hmi-p1", false], ["P1T1", "M230 245H315", "hmi-p1", true], ["P1T4", "M230 245V52H371A6 6 0 0 1 383 52H703V72", "hmi-p1", true],
  ["P2M", "M898 300H850V245", "hmi-p2", false], ["P2T2", "M850 245H765", "hmi-p2", true], ["P2T3", "M850 245V30H377V72", "hmi-p2", true],
  ["T3T1", "M377 172V198", "hmi-dr", true], ["T4T2", "M703 172V198", "hmi-dr", true], ["T1O", "M377 290V338", "hmi-dr", true], ["T2O", "M703 290V338", "hmi-dr", true],
]
const qtank: Screen = {
  /* splits and pump gains have their own readouts at the valves and pumps */
  cond: {},
  conds: (c, tr) => [tr("outlet restrictions T1–T4", "出口阻力 T1–T4"), [1, 2, 3, 4].map((j) => `${fx(c[`c${j}`], 2)}×`).join(" · ")],
  svg: (tr) => {
    const tank = (id: string) => {
      const [x, y, w, h] = QT.V[id], up = y < 150, lim = QT.lim[id]
      return `<rect class="hmi-vessel-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/>
      <clipPath id="qt-clip-${id}"><rect x="${x + 2}" y="${y + 2}" width="${w - 4}" height="${h - 4}" rx="6"/></clipPath>
      <g clip-path="url(#qt-clip-${id})"><rect class="hmi-liquid" data-liquid="${id}" x="${x}" y="${y + h}" width="${w}" height="${h}"/><line class="hmi-surface" data-surface="${id}" x1="${x}" x2="${x + w}" y1="${y + h}" y2="${y + h}"/>
      ${up ? "" : `<rect class="qt-band" data-band="${id}" x="${x}" y="0" width="${w}" height="0"/><line class="qt-sp" data-sp="${id}" x1="${x}" x2="${x + w}" y1="0" y2="0"/>`}</g>
      ${lim.map((l) => `<path class="hmi-limit" d="M${x + 1} ${qtY(id, l).toFixed(1)}h9M${x + w - 1} ${qtY(id, l).toFixed(1)}h-9"/>`).join("")}
      <rect class="hmi-vessel" data-vessel="${id}" x="${x}" y="${y}" width="${w}" height="${h}" rx="8"/>
      <text class="hmi-tag" x="${x + 10}" y="${y + 18}">${id}</text><text class="hmi-sub" x="${x + w - 10}" y="${y + 18}" text-anchor="end">${QT.h[id]}</text>
      <text class="hmi-value" data-value="${id}" x="${x + w / 2}" y="${y + h / 2 + (up ? 12 : 6)}" text-anchor="middle">—</text>
      ${up ? "" : `<text class="hmi-sub" data-sub="${id}" x="${x + w / 2}" y="${y + h / 2 + 24}" text-anchor="middle"></text>`}`
    }
    const pump = (id: string, cx: number, left: boolean) => {
      const tx = left ? cx - 38 : cx + 38, a = left ? "end" : "start"
      return `<circle class="hmi-pump running" cx="${cx}" cy="300" r="22"/><g data-rotor="${id}"><path class="hmi-rotor" d="M${cx} 300l0-14M${cx} 300l12.1 7M${cx} 300l-12.1 7"/><circle cx="${cx}" cy="300" r="3" class="hmi-hub"/></g>
      <text class="hmi-tag" x="${tx}" y="287" text-anchor="${a}">${id}</text><text class="hmi-value" data-value="${id}" x="${tx}" y="305" text-anchor="${a}">—</text>
      ${bar(id, left ? cx - 118 : cx + 38, 312)}<text class="hmi-sub" data-sub="${id}" x="${tx}" y="334" text-anchor="${a}"></text>`
    }
    return `<svg class="hmi-svg" viewBox="0 0 1080 384" role="img" aria-label="${esc(tr("Quadruple-tank process screen, replay", "四容水箱流程画面回放"))}">
  <defs>${arrowDef("qt-arrow")}</defs>
  <text class="hmi-note" x="512" y="22" text-anchor="middle">${tr("1 − γ2 of P2 → upper T3", "P2 的 1 − γ2 → 上水箱 T3")}</text><text class="hmi-note" x="512" y="45" text-anchor="middle">${tr("1 − γ1 of P1 → upper T4", "P1 的 1 − γ1 → 上水箱 T4")}</text>
  ${QT_PIPES.map(([id, d, c, a]) => pipe(id, d, c, a ? "qt-arrow" : undefined)).join("")}
  <circle class="hmi-node" cx="230" cy="245" r="3.5"/><circle class="hmi-node" cx="850" cy="245" r="3.5"/>
  <text class="hmi-sub" x="246" y="237">γ1</text><text class="hmi-value" data-value="g1" x="296" y="237" text-anchor="end">—</text>
  <text class="hmi-sub" x="784" y="237">γ2</text><text class="hmi-value" data-value="g2" x="834" y="237" text-anchor="end">—</text>
  <path class="hmi-reservoir" d="M120 340V366H960V340"/><path class="hmi-reservoir-l" d="M122 348H958"/><text class="hmi-note" x="540" y="362" text-anchor="middle">${tr("COMMON RESERVOIR", "公共储水池")}</text>
  ${["T3", "T4", "T1", "T2"].map(tank).join("")}${pump("P1", 160, true)}${pump("P2", 920, false)}
  ${clockBlock(540, 108, "middle")}
  <text class="hmi-sub" data-region x="540" y="190" text-anchor="middle"></text>
  <text class="hmi-sub" data-conds x="24" y="120"></text><text class="hmi-sub" data-conds2 x="24" y="138"></text>
  </svg>`
  },
  update: (q, s, tr) => {
    for (const id of ["T1", "T2", "T3", "T4"]) {
      const h = s.v[QT.h[id]], y = qtY(id, h), [lo, hi] = QT.lim[id], viol = h < lo || h > hi
      q(`[data-liquid="${id}"]`)?.setAttribute("y", y.toFixed(2))
      const sf = q(`[data-surface="${id}"]`); sf?.setAttribute("y1", y.toFixed(2)); sf?.setAttribute("y2", y.toFixed(2))
      const r = s.ref[QT.h[id]], tracked = r != null, out = tracked && Math.abs(h - r) > 0.5
      q(`[data-vessel="${id}"]`)?.classList.toggle("warn", viol || out)
      setText(q, `[data-value="${id}"]`, `${fx(h, 2)} cm`, viol || out)
      if (tracked) {
        const b1 = qtY(id, r + 0.5), b2 = qtY(id, r - 0.5), band = q(`[data-band="${id}"]`), sp = q(`[data-sp="${id}"]`)
        band?.setAttribute("y", b1.toFixed(1)); band?.setAttribute("height", (b2 - b1).toFixed(1))
        sp?.setAttribute("y1", qtY(id, r).toFixed(1)); sp?.setAttribute("y2", qtY(id, r).toFixed(1))
        setText(q, `[data-sub="${id}"]`, tr(`target ${fx(r, 2)} ± 0.5`, `目标 ${fx(r, 2)} ± 0.5`))
      }
    }
    for (const [id, k, kk] of [["P1", "u0", "k1"], ["P2", "u1", "k2"]] as const) {
      setText(q, `[data-value="${id}"]`, `${fx(s.v[k], 2)} V`); setBar(q, id, s.v[k] / 10)
      setText(q, `[data-sub="${id}"]`, tr(`gain ${fx(s.cond[kk], 2)} cm³/Vs`, `增益 ${fx(s.cond[kk], 2)} cm³/Vs`))
    }
    setText(q, '[data-value="g1"]', fx(s.cond.g1, 2)); setText(q, '[data-value="g2"]', fx(s.cond.g2, 2))
    const g = s.cond.g1 + s.cond.g2
    setText(q, "[data-region]", g > 1 ? tr(`γ1 + γ2 = ${fx(g, 2)} > 1 · minimum phase`, `γ1 + γ2 = ${fx(g, 2)} > 1 · 最小相位`) : tr(`γ1 + γ2 = ${fx(g, 2)} < 1 · non-minimum phase`, `γ1 + γ2 = ${fx(g, 2)} < 1 · 非最小相位`))
  },
  flows: (s) => {
    const f1 = (s.cond.k1 * s.v.u0) / 33, f2 = (s.cond.k2 * s.v.u1) / 33, d = (h: number, c: number) => c * Math.sqrt(Math.max(h, 0) / QT.top)
    return { S1: f1, S2: f2, P1M: f1, P2M: f2, P1T1: f1 * s.cond.g1, P1T4: f1 * (1 - s.cond.g1), P2T2: f2 * s.cond.g2, P2T3: f2 * (1 - s.cond.g2),
      T3T1: d(s.v.h3, s.cond.c3), T4T2: d(s.v.h4, s.cond.c4), T1O: d(s.v.h1, s.cond.c1), T2O: d(s.v.h2, s.cond.c2) }
  },
  rotors: (s) => ({ P1: s.v.u0 / 10, P2: s.v.u1 / 10 }),
}

/* ---------- Shell heavy oil fractionator ---------- */
const SH_IND: Record<string, number[]> = { y1: [850, 84, 190], y2: [850, 204, 190] }
const shell: Screen = {
  /* d1 and d2 have their own readouts at the exchangers; the gain uncertainty is shown whole */
  cond: {},
  conds: (c, tr) => [tr("gain uncertainty", "增益不确定性"), `ε = (${[1, 2, 3, 4, 5].map((j) => sgn(c[`e${j}`] ?? 0, 1)).join(", ")})`],
  svg: (tr) => {
    const ti = (k: string, y: number, label: string) => `<text class="hmi-sub" x="482" y="${y}">${label}</text><text class="hmi-value" data-value="${k}" x="580" y="${y}" text-anchor="end">—</text>`
    const hx = (cx: number, cy: number) => `<circle class="hmi-vessel-bg" cx="${cx}" cy="${cy}" r="17"/><circle class="hmi-vessel" cx="${cx}" cy="${cy}" r="17"/><path class="tep-coil" d="M${cx - 13} ${cy}H${cx - 7}L${cx - 2} ${cy - 9} ${cx + 3} ${cy + 9} ${cx + 7} ${cy}H${cx + 13}"/>`
    /* a pumparound: drawn off the column below, through the exchanger, returned above */
    const loop = (id: string, y: number, cls: string, base = "hmi-pipe") => pipe(`${id}o`, `M470 ${y + 10}H408`, cls, undefined, base) + pipe(`${id}r`, `M406 ${y - 10}H468`, cls, "sh-arrow", base)
    return `<svg class="hmi-svg" viewBox="0 0 1080 384" role="img" aria-label="${esc(tr("Shell fractionator process screen, replay", "Shell 分馏塔流程画面回放"))}">
  <defs>${arrowDef("sh-arrow")}</defs>
  ${pipe("feed", "M200 340H468", "hmi-dr", "sh-arrow")}<text class="hmi-tag" x="200" y="330">${tr("Feed", "进料")}</text>
  ${pipe("bot", "M530 352V372H1010", "hmi-dr", "sh-arrow")}<text class="hmi-tag" x="1010" y="364" text-anchor="end">${tr("Bottoms", "塔底产品")}</text>
  ${pipe("top", "M592 56H1010", "sh-u1", "sh-arrow")}${pipe("side", "M592 176H1010", "sh-u2", "sh-arrow")}
  ${loop("d2", 124, "hmi-dr", "ph-upset")}${loop("d1", 224, "hmi-dr")}${loop("u3", 296, "sh-u3")}
  ${hx(392, 124)}${hx(392, 224)}${hx(392, 296)}
  <rect class="hmi-vessel-bg" x="470" y="20" width="120" height="332" rx="30"/>
  <path class="tep-tray" d="M482 76H578M482 148H578M482 200H578M482 260H578M482 318H578" stroke-dasharray="4 4"/>
  <rect class="hmi-vessel" data-vessel="col" x="470" y="20" width="120" height="332" rx="30"/>
  ${ti("y3", 52, "y3")}${ti("y4", 116, "y4")}${ti("y5", 184, "y5")}${ti("y6", 232, "y6")}${ti("y7", 290, "y7")}<text class="hmi-sub" x="580" y="308" text-anchor="end">≥ −0.5</text>
  <text class="hmi-sub" x="530" y="332" text-anchor="middle">${tr("temperatures", "温度")}</text>
  <path class="ph-valve" d="M632 47 650 65V47L632 65Z"/><path class="ph-valve" d="M632 167 650 185V167L632 185Z"/>
  <text class="hmi-tag" x="664" y="44">${tr("u1 top draw", "u1 塔顶抽出")}</text><text class="hmi-value" data-value="u0" x="824" y="44" text-anchor="end">—</text>${bar("u0", 664, 66)}
  <text class="hmi-tag" x="664" y="164">${tr("u2 side draw", "u2 侧线抽出")}</text><text class="hmi-value" data-value="u1" x="824" y="164" text-anchor="end">—</text>${bar("u1", 664, 186)}
  <path d="M800 56V84" stroke="#9aa0a6" stroke-dasharray="3 3"/><circle class="ph-sensor" data-at="y1" cx="800" cy="100" r="16"/><text x="800" y="104" text-anchor="middle" class="hmi-sub">AT</text>
  <path d="M800 176V204" stroke="#9aa0a6" stroke-dasharray="3 3"/><circle class="ph-sensor" data-at="y2" cx="800" cy="220" r="16"/><text x="800" y="224" text-anchor="middle" class="hmi-sub">AT</text>
  ${indicator("y1", 850, 84, 190, tr("y1 top end point", "y1 塔顶端点"))}${indicator("y2", 850, 204, 190, tr("y2 side end point", "y2 侧线端点"))}
  <text class="hmi-tag" x="368" y="118" text-anchor="end">${tr("d2 upper reflux duty", "d2 上回流热负荷")}</text><text class="hmi-value" data-value="d2" x="368" y="136" text-anchor="end">—</text>
  <text class="hmi-sub" x="368" y="152" text-anchor="end">${tr("unmeasured, not seen by the controller", "不可测，控制器看不到")}</text>
  <text class="hmi-tag" x="368" y="218" text-anchor="end">${tr("d1 intermediate reflux duty", "d1 中间回流热负荷")}</text><text class="hmi-value" data-value="d1" x="368" y="236" text-anchor="end">—</text>
  <text class="hmi-sub" x="368" y="252" text-anchor="end">${tr("measured disturbance", "可测扰动")}</text>
  <text class="hmi-tag" x="368" y="290" text-anchor="end">${tr("u3 bottoms reflux duty", "u3 底回流热负荷")}</text><text class="hmi-value" data-value="u2" x="368" y="308" text-anchor="end">—</text>${bar("u2", 288, 314)}
  ${clockBlock(40, 46)}
  <text class="hmi-sub" data-conds x="620" y="300"></text><text class="hmi-sub" data-conds2 x="620" y="318"></text>
  </svg>`
  },
  update: (q, s, tr) => {
    for (const k of ["y3", "y4", "y5", "y6"]) setText(q, `[data-value="${k}"]`, sgn(s.v[k], 3))
    const low = s.v.y7 < -0.5
    setText(q, '[data-value="y7"]', sgn(s.v.y7, 3), low); q('[data-vessel="col"]')?.classList.toggle("warn", low || Math.abs(s.v.y1) > 0.5 || Math.abs(s.v.y2) > 0.5)
    for (const k of ["y1", "y2"]) {
      const [x, y, w] = SH_IND[k], r = s.ref[k]
      setIndicator(q, k, x, w, y, s.v[k], r, 0.02, sgn(s.v[k], 3), tr(`reference ${sgn(r, 3)} ± 0.02`, `参考 ${sgn(r, 3)} ± 0.02`))
      q(`[data-at="${k}"]`)?.classList.toggle("warn", Math.abs(s.v[k] - r) > 0.02)
    }
    for (const k of ["u0", "u1", "u2"]) { setText(q, `[data-value="${k}"]`, sgn(s.v[k], 3)); setBar(q, k, s.v[k] + 0.5) }
    setText(q, '[data-value="d1"]', sgn(s.v.d1, 2)); setText(q, '[data-value="d2"]', sgn(s.cond.d2 ?? 0, 2))
  },
  flows: (s) => {
    const f = (u: number) => 0.25 + 0.5 * (u + 0.5)
    return { feed: 0.6, bot: 0.4, top: f(s.v.u0), side: f(s.v.u1), u3o: f(s.v.u2), u3r: f(s.v.u2), d1o: f(s.v.d1), d1r: f(s.v.d1), d2o: f(s.cond.d2 ?? 0), d2r: f(s.cond.d2 ?? 0) }
  },
}

/* ---------- IWA BSM1: five reactors, settler, internal and sludge recycles ---------- */
const BS = { x: [150, 284, 418, 552, 686], y: 150, w: 120, h: 112 }
const BS_IND: Record<string, number> = { sno2: 1, so3: 2, so4: 3, so5: 4 }
const bsm: Screen = {
  cond: {
    temp: [["temperature", "水温"], (v) => `${v.toFixed(1)} °C`], alpha: [["aeration efficiency", "曝气效率"], pct], load: [["organic load", "有机负荷"], pct],
    nh: [["ammonia load", "氨氮负荷"], pct], qs: [["influent flow", "进水流量"], pct], v0s: [["settling velocity", "沉降速度"], pct],
  },
  svg: (tr) => {
    const reactor = (i: number) => {
      const x = BS.x[i], aer = i >= 2
      return `<rect class="${aer ? "bsm-aer" : "bsm-anox"}" x="${x}" y="${BS.y}" width="${BS.w}" height="${BS.h}" rx="8"/><rect class="hmi-vessel" data-vessel="R${i + 1}" x="${x}" y="${BS.y}" width="${BS.w}" height="${BS.h}" rx="8"/>
      <text class="hmi-tag" x="${x + 10}" y="${BS.y + 18}">R${i + 1}</text><text class="hmi-sub" x="${x + BS.w - 10}" y="${BS.y + 18}" text-anchor="end">${aer ? tr("aerobic", "好氧") : tr("anoxic", "缺氧")}</text>
      ${aer ? `<path class="bsm-diffuser" d="M${x + 16} ${BS.y + BS.h - 10}H${x + BS.w - 16}"/>${[0, 1, 2, 3, 4, 5].map((k) => `<circle class="bsm-bubble" data-bub="${i}-${k}" cx="${x + 24 + k * 14.4}" cy="${BS.y + BS.h - 14}" r="2.6"/>`).join("")}
        <text class="hmi-tag" x="${x + 10}" y="${BS.y + BS.h + 22}">KLa${i + 1}</text><text class="hmi-value" data-value="u${i - 2}" x="${x + BS.w - 6}" y="${BS.y + BS.h + 22}" text-anchor="end">—</text>${bar(`u${i - 2}`, x + 10, BS.y + BS.h + 30, 104)}` : ""}`
    }
    const ind = (name: string, i: number) => indicator(name, BS.x[i] + 10, BS.y + 36, 100, name.toUpperCase())
    return `<svg class="hmi-svg" viewBox="0 0 1080 384" role="img" aria-label="${esc(tr("BSM1 process screen, replay", "BSM1 流程画面回放"))}">
  <defs>${arrowDef("bs-arrow")}</defs>
  ${pipe("in", "M24 206H148", "hmi-dr", "bs-arrow")}
  <text class="hmi-tag" x="24" y="230">${tr("Influent", "进水")}</text><text class="hmi-value" data-value="q_in" x="24" y="248">—</text>
  <text class="hmi-sub" x="24" y="266">${tr("monitored", "只监视")}</text>
  ${pipe("qa", "M746 150V112H210V148", "bsm-qa", "bs-arrow")}
  <text class="hmi-tag" x="300" y="102">${tr("Qa internal recycle", "Qa 内回流")}</text><text class="hmi-value" data-value="u3" x="560" y="102" text-anchor="end">—</text>${bar("u3", 572, 97, 96)}
  ${[0, 1, 2, 3].map((i) => `<path class="hmi-pipe" d="M${BS.x[i] + BS.w} 206H${BS.x[i + 1] - 2}" marker-end="url(#bs-arrow)"/>`).join("")}
  ${pipe("to-st", "M806 206H844", "hmi-dr", "bs-arrow")}
  ${[0, 1, 2, 3, 4].map(reactor).join("")}
  ${ind("sno2", 1)}${ind("so3", 2)}${ind("so4", 3)}${ind("so5", 4)}
  <path class="hmi-vessel-bg" d="M846 150H986V244L930 272H902L846 244Z"/><path class="tep-tray" d="M856 168H976M856 184H976M856 200H976M856 216H976M856 232H976" stroke-dasharray="4 4"/>
  <path class="hmi-vessel" d="M846 150H986V244L930 272H902L846 244Z"/><text class="hmi-sub" x="936" y="292">${tr("settler, 10 layers", "二沉池，10 层")}</text>
  ${pipe("eff", "M986 160H1062", "hmi-dr", "bs-arrow")}
  <text class="hmi-tag" x="860" y="48">${tr("Effluent", "出水")}</text><text class="hmi-sub" x="1062" y="48" text-anchor="end">${tr("monitored, not scored", "只监视，不计分")}</text>
  <text class="hmi-sub" x="860" y="70">${tr("ammonia SNH", "氨氮 SNH")}</text><text class="hmi-value" data-value="snh_e" x="1062" y="70" text-anchor="end">—</text>
  <text class="hmi-sub" x="860" y="90">${tr("total nitrogen TN", "总氮 TN")}</text><text class="hmi-value" data-value="tn_e" x="1062" y="90" text-anchor="end">—</text>
  ${pipe("qr", "M916 272V318H210V264", "hmi-dr", "bs-arrow")}${pipe("qw", "M916 318H1062", "hmi-dr", "bs-arrow")}
  <text class="hmi-sub" x="540" y="336" text-anchor="middle">${tr("Qr return sludge 18,446 m³/d", "Qr 回流污泥 18,446 m³/天")}</text><text class="hmi-sub" x="1062" y="336" text-anchor="end">${tr("Qw waste 385 m³/d", "Qw 剩余污泥 385 m³/天")}</text>
  ${clockBlock(24, 44)}
  <text class="hmi-sub" data-conds x="24" y="360"></text><text class="hmi-sub" data-conds2 x="24" y="377"></text>
  </svg>`
  },
  update: (q, s, tr) => {
    for (const [k, i] of Object.entries(BS_IND)) {
      const band = k === "sno2" ? 0.4 : 0.3, r = s.ref[k], out = Math.abs(s.v[k] - r) > band
      setIndicator(q, k, BS.x[i] + 10, 100, BS.y + 36, s.v[k], r, band, fx(s.v[k], 2), tr(`target ${fx(r, 2)} ± ${band}`, `目标 ${fx(r, 2)} ± ${band}`))
      q(`[data-vessel="R${i + 1}"]`)?.classList.toggle("warn", out)
    }
    for (const k of ["u0", "u1", "u2"]) { setText(q, `[data-value="${k}"]`, `${Math.round(s.v[k])} /d`); setBar(q, k, s.v[k] / 360, 104) }
    setText(q, '[data-value="u3"]', `${number(Math.round(s.v.u3))} m³/d`); setBar(q, "u3", s.v.u3 / 92230, 96)
    setText(q, '[data-value="q_in"]', tr(`${number(Math.round(s.v.q_in * 1000))} m³/d`, `${number(Math.round(s.v.q_in * 1000))} m³/天`)); setText(q, '[data-value="snh_e"]', `${fx(s.v.snh_e, 2)} g N/m³`); setText(q, '[data-value="tn_e"]', `${fx(s.v.tn_e, 1)} g N/m³`)
  },
  flows: (s) => { const f = Math.min(s.v.q_in / 40, 1.5); return { in: f, "to-st": f, eff: f * 0.8, qa: s.v.u3 / 92230, qr: 0.35, qw: 0.15 } },
  bubbles: (s) => ({ 2: s.v.u0 / 360, 3: s.v.u1 / 360, 4: s.v.u2 / 360 }),
}

export const SCREENS: Record<string, Screen> = { qtank, shell, bsm }

/* The conditions line(s): what differs from the start of the scenario. */
export function condLines(sc: Screen, cond: Record<string, number>, start: Record<string, number>, tr: Tr): [string, string] {
  const moved = Object.keys(sc.cond).filter((k) => cond[k] != null && start[k] != null && Math.abs(cond[k] - start[k]) > 1e-9)
  if (!moved.length) return [tr("Conditions: as at the start", "工况：与起点相同"), ""]
  const parts = moved.map((k) => `${tr(sc.cond[k][0][0], sc.cond[k][0][1])} ${sc.cond[k][1](cond[k])}`), half = Math.ceil(parts.length / 2), sep = tr(", ", "，")
  const head = tr("Since the start: ", "相对起点：")
  return parts.length > 2 ? [head + parts.slice(0, half).join(sep), parts.slice(half).join(sep)] : [head + parts.join(sep), ""]
}
