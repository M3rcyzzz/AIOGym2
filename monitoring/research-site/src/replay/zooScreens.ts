import type { Tr } from "@/lib/i18n"
import { esc, fx, number } from "@/lib/format"
import { arrowDef, bar, clockBlock, indicator, pct, pipe, setBar, setIndicator, setText, type Layout, type NetFlow, type Q, type Screen, type Snap } from "./suiteScreens"

/* Process screens of the plant-zoo families (docs/plant-zoo.md), in the vocabulary of suiteScreens.ts: neutral
   equipment, pipes whose dashes move with the flow, MV bars, indicators with the band around the reference for
   tracked outputs, readouts that turn orange outside their limits. Topology and tags follow the environment code
   in experiments/scenario_ppo/zoo/*.py and proc_batch.py. MV pipes carry the MV colours of the trends (blue,
   purple, red-brown in MV order). */

const f1 = (u: string) => (v: number) => `${fx(v, 1)}${u}`
const f2 = (u: string) => (v: number) => `${fx(v, 2)}${u}`
const f3 = (u: string) => (v: number) => `${fx(v, 3)}${u}`
const MVC = ["sh-u1", "sh-u2", "sh-u3", "hmi-p1", "hmi-p2"]
/* a valve on a horizontal (or vertical) pipe */
const valve = (x: number, y: number, vert = false) =>
  vert ? `<path class="ph-valve" d="M${x - 9} ${y - 9}H${x + 9}L${x - 9} ${y + 9}H${x + 9}Z"/>` : `<path class="ph-valve" d="M${x - 9} ${y - 9} ${x + 9} ${y + 9}V${y - 9}L${x - 9} ${y + 9}Z"/>`
/* an MV: tag and value on one line, its bar underneath */
const mv = (k: string, x: number, y: number, tag: string, w = 150) =>
  `<text class="hmi-tag" x="${x}" y="${y}">${tag}</text><text class="hmi-value" data-value="${k}" x="${x + w}" y="${y}" text-anchor="end">—</text>${bar(k, x, y + 8, w)}`
function setMv(q: Q, k: string, v: number, lo: number, hi: number, text: string, w = 150) { setText(q, `[data-value="${k}"]`, text); setBar(q, k, (v - lo) / (hi - lo), w) }
/* a readout: tag and value on one line, an optional fixed sub line */
const readout = (k: string, x: number, y: number, tag: string, w = 160, sub = "") =>
  `<text class="hmi-tag" x="${x}" y="${y}">${tag}</text><text class="hmi-value" data-value="${k}" x="${x + w}" y="${y}" text-anchor="end">—</text>${sub ? `<text class="hmi-sub" data-sub="${k}" x="${x}" y="${y + 18}">${sub}</text>` : ""}`
const sensor = (k: string, x: number, y: number, label: string, tap?: string) =>
  `${tap ? `<path d="${tap}" stroke="#9aa0a6" stroke-dasharray="3 3"/>` : ""}<circle class="ph-sensor" data-at="${k}" cx="${x}" cy="${y}" r="15"/><text x="${x}" y="${y + 4}" text-anchor="middle" class="hmi-sub">${label}</text>`
const hxc = (cx: number, cy: number, r = 18) =>
  `<circle class="hmi-vessel-bg" cx="${cx}" cy="${cy}" r="${r}"/><circle class="hmi-vessel" cx="${cx}" cy="${cy}" r="${r}"/><path class="tep-coil" d="M${cx - r + 4} ${cy}H${cx - 7}L${cx - 2} ${cy - 9} ${cx + 3} ${cy + 9} ${cx + 7} ${cy}H${cx + r - 4}"/>`
const agitator = (cx: number, y0: number, y1: number) => `<path class="tep-agitator" d="M${cx} ${y0}V${y1}M${cx - 26} ${y1}H${cx + 26}M${cx - 26} ${y1 - 7}V${y1 + 7}M${cx + 26} ${y1 - 7}V${y1 + 7}"/>`
/* a stirred vessel with a jacket around its lower part and fixed liquid content */
function cstr(id: string, x: number, y: number, w: number, h: number, jacket: boolean, tag: string) {
  const lq = y + h * 0.22
  return `<rect class="hmi-vessel-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="26"/>
  <clipPath id="z-clip-${id}"><rect x="${x + 2}" y="${y + 2}" width="${w - 4}" height="${h - 4}" rx="24"/></clipPath>
  <g clip-path="url(#z-clip-${id})"><rect class="hmi-liquid" x="${x}" y="${lq}" width="${w}" height="${h}"/><line class="hmi-surface" x1="${x}" x2="${x + w}" y1="${lq}" y2="${lq}"/></g>
  ${jacket ? `<path class="hmi-vessel" d="M${x - 12} ${y + h * 0.3}V${y + h - 8}Q${x - 12} ${y + h + 12} ${x + 14} ${y + h + 12}H${x + w - 14}Q${x + w + 12} ${y + h + 12} ${x + w + 12} ${y + h - 8}V${y + h * 0.3}"/>` : ""}
  <rect class="hmi-vessel" data-vessel="${id}" x="${x}" y="${y}" width="${w}" height="${h}" rx="26"/>${agitator(x + w / 2, y - 18, y + h * 0.72)}
  <text class="hmi-tag" x="${x + 14}" y="${y + 20}">${tag}</text>`
}
/* outside the limits: value and its sensor turn orange */
function limited(q: Q, k: string, v: number, lo: number | null, hi: number | null, text: string) {
  const bad = (lo != null && v < lo) || (hi != null && v > hi)
  setText(q, `[data-value="${k}"]`, text, bad); q(`[data-at="${k}"]`)?.classList.toggle("warn", bad)
  return bad
}
type Ind = Record<string, [number, number, number]>   // x, y, width of each indicator
function tracked(q: Q, s: Snap, tr: Tr, at: Ind, k: string, band: number, dec: number, unit: string) {
  const [x, y, w] = at[k], v = s.v[k], r = s.ref[k], u = unit ? ` ${unit}` : ""
  setIndicator(q, k, x, w, y, v, r, band, `${fx(v, dec)}${u}`, tr(`target ${fx(r, dec)} ± ${band}`, `目标 ${fx(r, dec)} ± ${band}`))
  const out = Math.abs(v - r) > band
  q(`[data-at="${k}"]`)?.classList.toggle("warn", out)
  return out
}
const inds = (at: Ind, tags: Record<string, string>) => Object.entries(at).map(([k, [x, y, w]]) => indicator(k, x, y, w, tags[k])).join("")
const svgOpen = (label: string, arrow: string) => `<svg class="hmi-svg" viewBox="0 0 1080 384" role="img" aria-label="${esc(label)}"><defs>${arrowDef(arrow)}</defs>`
const condsAt = (x: number, y: number, anchor = "start") => `<text class="hmi-sub" data-conds x="${x}" y="${y}" text-anchor="${anchor}"></text><text class="hmi-sub" data-conds2 x="${x}" y="${y + 18}" text-anchor="${anchor}"></text>`

/* ---------- Newell & Lee forced-circulation evaporator (evaporator.py) ---------- */
const EV: Ind = { P2: [900, 196, 150], X2: [900, 292, 150] }
const EV_SEP = { x: 430, y: 70, w: 140, h: 230, top: 2 }
const evY = (L: number) => EV_SEP.y + EV_SEP.h - 2 - (Math.min(Math.max(L, 0), EV_SEP.top) / EV_SEP.top) * (EV_SEP.h - 4)
const evap: Screen = {
  cond: {
    F1: [["feed flow", "进料流量"], f1(" kg/min")], X1: [["feed composition", "进料浓度"], f2(" %")], T1: [["feed temperature", "进料温度"], f1(" °C")],
    T200: [["cooling-water temperature", "冷却水温度"], f1(" °C")], ua1: [["calandria heat transfer", "加热器传热"], pct], ua2: [["condenser heat transfer", "冷凝器传热"], pct],
  },
  svg: (tr) => {
    const { x, y, w, h } = EV_SEP
    return `${svgOpen(tr("Evaporator process screen, replay", "蒸发器流程画面回放"), "ev-arrow")}
  ${pipe("feed", "M40 330H283", "hmi-dr")}<circle class="hmi-node" cx="285" cy="330" r="3.5"/>
  ${readout("F1", 40, 318, tr("Feed F1", "进料 F1"), 190)}<text class="hmi-sub" x="40" y="350">${tr("measured, fed forward", "可测，可作前馈")}</text>
  ${pipe("rec", "M470 300V330H285V302", "hmi-dr", "ev-arrow")}${pipe("rise", "M300 150V120H428", "hmi-dr", "ev-arrow")}
  ${pipe("steam", "M40 214H248", "sh-u2", "ev-arrow")}${valve(200, 214)}${mv("u1", 40, 186, tr("P100 steam", "P100 蒸汽"), 150)}
  <rect class="hmi-vessel-bg" x="250" y="150" width="70" height="152" rx="10"/><path class="tep-tray" d="M264 160V292M285 160V292M306 160V292"/>
  <rect class="hmi-vessel" x="250" y="150" width="70" height="152" rx="10"/><text class="hmi-tag" x="244" y="142" text-anchor="end">${tr("Calandria", "加热器")}</text>
  <rect class="hmi-vessel-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="24"/>
  <clipPath id="ev-clip"><rect x="${x + 2}" y="${y + 2}" width="${w - 4}" height="${h - 4}" rx="22"/></clipPath>
  <g clip-path="url(#ev-clip)"><rect class="hmi-liquid" data-liquid="L2" x="${x}" y="${y + h}" width="${w}" height="${h}"/><line class="hmi-surface" data-surface="L2" x1="${x}" x2="${x + w}" y1="${y + h}" y2="${y + h}"/>
  <rect class="qt-band" data-band="L2" x="${x}" y="0" width="${w}" height="0"/><line class="qt-sp" data-sp="L2" x1="${x}" x2="${x + w}" y1="0" y2="0"/></g>
  ${[0.3, 1.8].map((l) => `<path class="hmi-limit" d="M${x + 1} ${evY(l).toFixed(1)}h9M${x + w - 1} ${evY(l).toFixed(1)}h-9"/>`).join("")}
  <rect class="hmi-vessel" data-vessel="L2" x="${x}" y="${y}" width="${w}" height="${h}" rx="24"/>
  <text class="hmi-tag" x="${x + 14}" y="${y + 22}">${tr("Separator", "分离器")}</text><text class="hmi-sub" x="${x + w - 14}" y="${y + 22}" text-anchor="end">L2</text>
  <text class="hmi-value" data-value="L2" x="${x + w / 2}" y="${y + 52}" text-anchor="middle">—</text><text class="hmi-sub" data-sub="L2" x="${x + w / 2}" y="${y + 70}" text-anchor="middle"></text>
  ${pipe("prod", "M530 300V340H800", "sh-u1", "ev-arrow")}${valve(650, 340)}${mv("u0", 584, 366, tr("F2 product", "F2 产品"), 200)}
  ${sensor("X2", 760, 306, "AT", "M760 340V321")}
  ${pipe("vap", "M500 70V40H660V110H700", "hmi-dr", "ev-arrow")}${sensor("P2", 600, 70, "PT", "M600 40V55")}
  ${hxc(720, 110, 20)}<text class="hmi-tag" x="690" y="150" text-anchor="end">${tr("Condenser", "冷凝器")}</text>
  ${pipe("cond", "M720 130V176", "hmi-dr", "ev-arrow")}<text class="hmi-sub" x="712" y="176" text-anchor="end">${tr("condensate", "冷凝液")}</text>
  ${pipe("cw", "M860 110H742", "sh-u3", "ev-arrow")}${valve(820, 110)}${mv("u2", 736, 150, tr("F200 water", "F200 冷却水"), 180)}
  ${pipe("ret", "M720 90V40H880", "sh-u3", "ev-arrow")}${readout("T201", 736, 28, tr("T201 return ≤ 50 °C", "T201 回水 ≤ 50 °C"), 320)}
  ${inds(EV, { P2: tr("P2 pressure", "P2 压力"), X2: tr("X2 product", "X2 产品浓度") })}
  ${clockBlock(24, 40)}${condsAt(24, 104)}
  </svg>`
  },
  update: (q, s, tr) => {
    const L = s.v.L2, r = s.ref.L2, yL = evY(L), viol = L < 0.3 || L > 1.8, out = Math.abs(L - r) > 0.05
    q('[data-liquid="L2"]')?.setAttribute("y", yL.toFixed(2)); const sf = q('[data-surface="L2"]'); sf?.setAttribute("y1", yL.toFixed(2)); sf?.setAttribute("y2", yL.toFixed(2))
    const b1 = evY(r + 0.05), b2 = evY(r - 0.05), band = q('[data-band="L2"]'), sp = q('[data-sp="L2"]')
    band?.setAttribute("y", b1.toFixed(1)); band?.setAttribute("height", (b2 - b1).toFixed(1)); sp?.setAttribute("y1", evY(r).toFixed(1)); sp?.setAttribute("y2", evY(r).toFixed(1))
    q('[data-vessel="L2"]')?.classList.toggle("warn", viol || out)
    setText(q, '[data-value="L2"]', `${fx(L, 3)} m`, viol || out); setText(q, '[data-sub="L2"]', tr(`target ${fx(r, 2)} ± 0.05`, `目标 ${fx(r, 2)} ± 0.05`))
    tracked(q, s, tr, EV, "P2", 1, 1, "kPa"); tracked(q, s, tr, EV, "X2", 0.5, 2, "%")
    limited(q, "T201", s.v.T201, null, 50, `${fx(s.v.T201, 1)} °C`)
    setText(q, '[data-value="F1"]', `${fx(s.v.F1, 2)} kg/min`)
    setMv(q, "u0", s.v.u0, 0, 5, `${fx(s.v.u0, 2)} kg/min`, 200); setMv(q, "u1", s.v.u1, 100, 400, `${fx(s.v.u1, 0)} kPa`); setMv(q, "u2", s.v.u2, 0, 400, `${fx(s.v.u2, 0)} kg/min`, 180)
  },
  flows: (s) => ({ feed: s.v.F1 / 12, rec: 0.6, rise: 0.6, steam: s.v.u1 / 300, prod: s.v.u0 / 3, vap: 0.5, cond: 0.4, cw: s.v.u2 / 300, ret: s.v.u2 / 300 }),
}

/* ---------- Van de Vusse reactor (vdv.py) ---------- */
const VD: Ind = { cB: [880, 92, 170], T: [880, 182, 170] }
const vdv: Screen = {
  cond: { cA0: [["feed concentration", "进料浓度"], f2(" mol/l")], T0: [["feed temperature", "进料温度"], f1(" °C")], kw: [["jacket heat transfer", "夹套传热"], pct], act: [["catalyst activity", "催化剂活性"], pct] },
  svg: (tr) => `${svgOpen(tr("Van de Vusse reactor process screen, replay", "Van de Vusse 反应器流程画面回放"), "vd-arrow")}
  <text class="hmi-note" x="520" y="28" text-anchor="middle">A → B → C · 2A → D</text>
  ${pipe("feed", "M150 110H470V128", "sh-u1", "vd-arrow")}${valve(320, 110)}${mv("u0", 150, 82, tr("r dilution rate", "r 稀释率"), 170)}
  <text class="hmi-sub" data-value="feedc" x="150" y="134"></text>
  ${cstr("R", 440, 130, 160, 170, true, tr("CSTR", "反应釜"))}
  ${sensor("T", 560, 230, "TT")}
  ${pipe("jk", "M360 290H426", "sh-u2", "vd-arrow")}${mv("u1", 150, 262, tr("QK jacket heat", "QK 夹套热量"), 200)}
  <text class="hmi-sub" x="150" y="306">${tr("negative = heat removed", "负值为移出热量")}</text>
  ${readout("TK", 440, 340, tr("TK jacket", "TK 夹套"), 160)}<text class="hmi-sub" x="440" y="358">${tr("monitored", "只监视")}</text>
  ${pipe("prod", "M600 250H860", "hmi-dr", "vd-arrow")}${sensor("cB", 720, 226, "AT", "M720 250V241")}
  ${readout("cA", 880, 262, tr("cA reactant ≥ 0.8", "cA 反应物 ≥ 0.8"), 170)}<text class="hmi-sub" x="880" y="280">mol/l</text>
  ${inds(VD, { cB: tr("cB product", "cB 产物"), T: tr("T reactor", "T 反应温度") })}
  ${clockBlock(1056, 36, "end")}${condsAt(24, 40)}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, VD, "cB", 0.02, 3, "mol/l"), o2 = tracked(q, s, tr, VD, "T", 0.5, 1, "°C")
    const lo = limited(q, "cA", s.v.cA, 0.8, null, `${fx(s.v.cA, 3)}`)
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || lo || s.v.T > 140)
    setText(q, '[data-value="TK"]', `${fx(s.v.TK, 1)} °C`)
    setText(q, '[data-value="feedc"]', tr(`feed cA0 ${fx(s.cond.cA0, 2)} mol/l · T0 ${fx(s.cond.T0, 1)} °C`, `进料 cA0 ${fx(s.cond.cA0, 2)} mol/l · T0 ${fx(s.cond.T0, 1)} °C`))
    setMv(q, "u0", s.v.u0, 3, 35, `${fx(s.v.u0, 2)} /h`, 170); setMv(q, "u1", -s.v.u1, 0, 9000, `${number(Math.round(s.v.u1))} kJ/h`, 200)
  },
  flows: (s) => ({ feed: s.v.u0 / 25, prod: s.v.u0 / 25, jk: -s.v.u1 / 5000 }),
}

/* ---------- exothermic CSTR under its base-layer temperature loop (excstr.py) ---------- */
const EX: Ind = { CA: [870, 96, 180], prod: [870, 186, 180] }
const excstr: Screen = {
  cond: { CAf: [["feed concentration", "进料浓度"], f2(" mol/l")], Tf: [["feed temperature", "进料温度"], f1(" K")], ua: [["jacket heat transfer", "夹套传热"], pct], act: [["catalyst activity", "催化剂活性"], pct] },
  svg: (tr) => `${svgOpen(tr("Exothermic CSTR process screen, replay", "放热 CSTR 流程画面回放"), "ex-arrow")}
  <text class="hmi-note" x="520" y="28" text-anchor="middle">${tr("A → B, exothermic · open-loop unstable", "A → B，放热 · 开环不稳定")}</text>
  ${pipe("feed", "M150 110H480V128", "sh-u2", "ex-arrow")}${valve(330, 110)}${mv("u1", 150, 82, tr("q feed", "q 进料"), 170)}
  <text class="hmi-sub" data-value="feedc" x="150" y="134"></text>
  ${cstr("R", 450, 130, 160, 170, true, tr("CSTR", "反应釜"))}
  <text class="hmi-sub" x="464" y="206">${tr("T reactor", "T 反应温度")}</text><text class="hmi-value" data-value="T" x="464" y="224">—</text>
  <circle class="hmi-vessel-bg" cx="380" cy="220" r="18"/><circle class="hmi-vessel" cx="380" cy="220" r="18"/><text class="hmi-sub" x="380" y="224" text-anchor="middle">TC</text>
  <path class="tep-util" d="M398 220H448"/><path class="tep-util" d="M380 238V331"/>
  <path class="tep-util sh-u1" d="M320 220H362"/>${mv("u0", 150, 206, tr("Tsp setpoint", "Tsp 温度设定"), 170)}
  <text class="hmi-sub" x="150" y="248">${tr("base-layer PI moves the coolant", "基础层 PI 调冷却剂")}</text>
  ${pipe("cool", "M300 340H530V314", "hmi-dr", "ex-arrow")}${valve(380, 340)}${readout("Tc", 150, 330, tr("Tc coolant", "Tc 冷却剂"), 140)}<text class="hmi-sub" x="150" y="348">${tr("limits 270–330 K", "限值 270–330 K")}</text>
  ${pipe("prod", "M610 250H850", "hmi-dr", "ex-arrow")}${sensor("CA", 720, 226, "AT", "M720 250V241")}
  ${inds(EX, { CA: tr("CA outlet", "CA 出口浓度"), prod: tr("Production", "产量") })}
  ${clockBlock(1056, 36, "end")}${condsAt(1056, 330, "end")}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, EX, "CA", 0.01, 3, "mol/l"), o2 = tracked(q, s, tr, EX, "prod", 0.5, 1, "mol/min")
    const lim = limited(q, "Tc", s.v.Tc, 270, 330, `${fx(s.v.Tc, 1)} K`)
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || lim)
    setText(q, '[data-value="T"]', `${fx(s.v.T, 1)} K`)
    setText(q, '[data-value="feedc"]', tr(`feed CAf ${fx(s.cond.CAf, 2)} mol/l · Tf ${fx(s.cond.Tf, 1)} K`, `进料 CAf ${fx(s.cond.CAf, 2)} mol/l · Tf ${fx(s.cond.Tf, 1)} K`))
    setMv(q, "u0", s.v.u0, 330, 375, `${fx(s.v.u0, 1)} K`, 170); setMv(q, "u1", s.v.u1, 60, 140, `${fx(s.v.u1, 1)} l/min`, 170)
  },
  flows: (s) => ({ feed: s.v.u1 / 120, prod: s.v.u1 / 120, cool: Math.max(0.15, (340 - s.v.Tc) / 70) }),
}

/* ---------- continuous bioreactor (bioreactor.py) ---------- */
const BI: Ind = { X: [880, 92, 170], P: [880, 182, 170] }
const bio: Screen = {
  cond: {
    mum: [["maximum growth rate", "最大比生长速率"], f3(" 1/h")], Km: [["saturation constant", "饱和常数"], f3(" g/l")], K1: [["inhibition constant", "抑制常数"], f3(" l/g")],
    Y: [["yield", "得率"], f3("")], alpha: [["product coefficient", "产物系数"], f2("")],
  },
  svg: (tr) => `${svgOpen(tr("Bioreactor process screen, replay", "生物反应器流程画面回放"), "bi-arrow")}
  <text class="hmi-note" x="535" y="28" text-anchor="middle">${tr("substrate → biomass + product · washout if D is too high", "底物 → 菌体 + 产物 · D 过高会冲出")}</text>
  ${pipe("feed", "M150 110H500V128", "sh-u1", "bi-arrow")}${valve(390, 110)}${mv("u0", 150, 82, tr("D dilution rate", "D 稀释率"), 190)}
  ${pipe("conc", "M250 196V112", "sh-u2", "bi-arrow")}<circle class="hmi-node" cx="250" cy="110" r="3.5"/>${mv("u1", 150, 214, tr("Sf feed substrate", "Sf 进料底物"), 190)}
  ${cstr("R", 455, 130, 160, 180, false, tr("Fermenter", "发酵罐"))}
  ${pipe("prod", "M615 262H860", "hmi-dr", "bi-arrow")}${sensor("X", 720, 238, "AT", "M720 262V253")}
  ${readout("S", 880, 262, tr("S substrate ≤ 0.4", "S 底物 ≤ 0.4"), 170)}<text class="hmi-sub" x="880" y="280">${tr("g/l, above it growth is inhibited", "g/l，超过则生长受抑制")}</text>
  ${inds(BI, { X: tr("X biomass", "X 菌体"), P: tr("P product", "P 产物") })}
  ${clockBlock(1056, 36, "end")}${condsAt(24, 340)}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, BI, "X", 0.02, 3, "g/l"), o2 = tracked(q, s, tr, BI, "P", 0.1, 2, "g/l")
    const lim = limited(q, "S", s.v.S, null, 0.4, `${fx(s.v.S, 3)}`)
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || lim || s.v.X < 0.01)
    setMv(q, "u0", s.v.u0, 0.05, 0.6, `${fx(s.v.u0, 3)} /h`, 190); setMv(q, "u1", s.v.u1, 1.5, 8, `${fx(s.v.u1, 2)} g/l`, 190)
  },
  flows: (s) => ({ feed: s.v.u0 / 0.45, conc: s.v.u0 / 0.6, prod: s.v.u0 / 0.45 }),
}

/* ---------- Skogestad's column A (column.py) ---------- */
const CO: Ind = { yD: [880, 120, 170], yB: [880, 300, 170] }
const column: Screen = {
  cond: { F: [["feed rate", "进料量"], f2(" kmol/min")], zF: [["feed composition", "进料组成"], f2("")], qF: [["feed quality", "进料热状况"], f2("")], alpha: [["relative volatility", "相对挥发度"], f2("")] },
  svg: (tr) => `${svgOpen(tr("Distillation column process screen, replay", "精馏塔流程画面回放"), "co-arrow")}
  <rect class="hmi-vessel-bg" x="480" y="50" width="100" height="290" rx="30"/>
  <path class="tep-tray" d="${[90, 120, 150, 180, 212, 244, 276, 306].map((y) => `M490 ${y}H570`).join("")}" stroke-dasharray="4 4"/>
  <rect class="hmi-vessel" x="480" y="50" width="100" height="290" rx="30"/><text class="hmi-sub" x="530" y="200" text-anchor="middle">${tr("41 stages", "41 级")}</text>
  ${pipe("vap", "M530 50V40H630", "hmi-dr", "co-arrow")}${hxc(650, 40)}<text class="hmi-sub" x="676" y="26">${tr("condenser", "冷凝器")}</text>
  ${pipe("top", "M668 40H720V80", "hmi-dr")}<circle class="hmi-node" cx="720" cy="80" r="3.5"/>
  ${pipe("lt", "M720 80H582", "sh-u1", "co-arrow")}${valve(650, 80)}${mv("u0", 596, 106, tr("LT reflux", "LT 回流"), 110)}
  ${pipe("dist", "M720 80V150H860", "hmi-dr", "co-arrow")}${sensor("yD", 800, 126, "AT", "M800 150V141")}<text class="hmi-tag" x="860" y="170" text-anchor="end">${tr("Distillate", "塔顶产品")}</text>
  ${pipe("liq", "M530 340V352H630", "hmi-dr", "co-arrow")}${hxc(650, 352)}<text class="hmi-sub" x="690" y="380">${tr("reboiler", "再沸器")}</text>
  ${pipe("vb", "M650 334V310H582", "sh-u2", "co-arrow")}${mv("u1", 596, 270, tr("VB boilup", "VB 再沸"), 110)}
  ${pipe("bot", "M668 352H860", "hmi-dr", "co-arrow")}${sensor("yB", 800, 328, "AT", "M800 352V343")}<text class="hmi-tag" x="860" y="372" text-anchor="end">${tr("Bottoms", "塔底产品")}</text>
  ${pipe("feed", "M300 196H478", "hmi-dr", "co-arrow")}${readout("F", 300, 184, tr("Feed F", "进料 F"), 160)}<text class="hmi-sub" data-value="feedc" x="300" y="216"></text>
  <path class="tep-util" d="M440 110H478"/>${readout("x31", 300, 114, tr("Stage 31 x", "第 31 级 x"), 130)}
  <path class="tep-util" d="M440 290H478"/>${readout("x10", 300, 294, tr("Stage 10 x", "第 10 级 x"), 130)}<text class="hmi-sub" x="300" y="312">${tr("monitored", "只监视")}</text>
  ${inds(CO, { yD: tr("yD top purity", "yD 塔顶纯度"), yB: tr("yB bottom purity", "yB 塔底纯度") })}
  <text class="hmi-sub" x="880" y="232">${tr("purity as −log₁₀ impurity", "纯度以 −log₁₀ 杂质计")}</text>
  ${clockBlock(24, 40)}${condsAt(24, 350)}
  </svg>`,
  update: (q, s, tr) => {
    tracked(q, s, tr, CO, "yD", 0.05, 3, ""); tracked(q, s, tr, CO, "yB", 0.05, 3, "")
    setText(q, '[data-value="F"]', `${fx(s.v.F, 3)} kmol/min`); setText(q, '[data-value="x31"]', fx(s.v.x31, 3)); setText(q, '[data-value="x10"]', fx(s.v.x10, 3))
    setText(q, '[data-value="feedc"]', tr(`zF ${fx(s.cond.zF, 2)} · qF ${fx(s.cond.qF, 2)}`, `zF ${fx(s.cond.zF, 2)} · qF ${fx(s.cond.qF, 2)}`))
    setMv(q, "u0", s.v.u0, 2.2, 3.2, fx(s.v.u0, 3), 110); setMv(q, "u1", s.v.u1, 2.7, 3.7, fx(s.v.u1, 3), 110)
  },
  flows: (s) => ({ vap: 0.5, top: 0.5, lt: s.v.u0 / 3, dist: 0.25, liq: 0.5, vb: s.v.u1 / 3.5, bot: 0.25, feed: s.v.F / 1.2 }),
}

/* ---------- heat-exchanger train with bypass and trim cooler (hexnet.py) ---------- */
const HX: Ind = { Tc_out: [150, 284, 180], Thc: [880, 160, 170] }
const hex: Screen = {
  cond: {
    Fh: [["hot flow", "热物流流量"], f1(" kg/s")], Th_in: [["hot inlet temperature", "热物流入口温度"], f1(" °C")], Fc: [["cold flow", "冷物流流量"], f1(" kg/s")],
    Tc_in: [["cold inlet temperature", "冷物流入口温度"], f1(" °C")], ua: [["exchanger heat transfer", "换热器传热"], pct], Tw_in: [["water temperature", "冷却水温度"], f1(" °C")],
  },
  svg: (tr) => `${svgOpen(tr("Heat-exchanger train process screen, replay", "换热网络流程画面回放"), "hx-arrow")}
  ${pipe("hin", "M24 130H198", "hmi-dr")}<circle class="hmi-node" cx="200" cy="130" r="3.5"/>${readout("Th_in", 24, 118, tr("Hot in", "热物流入"), 160)}
  <text class="hmi-sub" data-value="hotc" x="24" y="154"></text>
  ${pipe("byp", "M200 130V70H700V128", "sh-u1", "hx-arrow")}${valve(450, 70)}${mv("u0", 380, 40, tr("b bypass", "b 旁路"), 140)}
  ${pipe("tube", "M200 130H700", "hmi-dr")}<circle class="hmi-node" cx="700" cy="130" r="3.5"/>
  <rect class="hmi-vessel-bg" x="300" y="108" width="300" height="44" rx="20"/><path class="tep-coil" d="M310 130H340L350 120 360 140 370 120 380 140 390 120 400 140 410 120 420 140 430 120 440 140 450 120 460 140 470 120 480 140 490 120 500 140 510 120 520 140 530 120 540 140 550 120 560 140H590"/>
  <rect class="hmi-vessel" data-vessel="E" x="300" y="108" width="300" height="44" rx="20"/><text class="hmi-sub" x="450" y="170" text-anchor="middle">${tr("process exchanger, counter-current", "工艺换热器，逆流")}</text>
  ${pipe("cin", "M580 260V154", "hmi-dr", "hx-arrow")}<text class="hmi-tag" x="590" y="250">${tr("Cold in", "冷物流入")}</text><text class="hmi-sub" data-value="coldc" x="590" y="268"></text>
  ${pipe("cout", "M320 154V250H150", "hmi-dr", "hx-arrow")}${sensor("Tc_out", 260, 226, "TT", "M260 250V241")}
  ${pipe("mix", "M700 130H758", "hmi-dr", "hx-arrow")}${hxc(780, 130, 20)}<text class="hmi-sub" x="756" y="172" text-anchor="end">${tr("trim cooler", "后冷器")}</text>
  ${pipe("hout", "M802 130H1056", "hmi-dr", "hx-arrow")}${sensor("Thc", 960, 106, "TT", "M960 130V121")}
  ${pipe("cw", "M780 300V152", "sh-u2", "hx-arrow")}${valve(780, 250, true)}${mv("u1", 800, 260, tr("Fw cooling water", "Fw 冷却水"), 230)}
  ${pipe("ret", "M790 112V40H1056", "hmi-dr", "hx-arrow")}${readout("Tw", 880, 28, tr("Tw return ≤ 45 °C", "Tw 回水 ≤ 45 °C"), 170)}
  ${inds(HX, { Tc_out: tr("Tc,out cold outlet", "Tc,out 冷物流出口"), Thc: tr("Thc hot product", "Thc 热物流出口") })}
  ${clockBlock(24, 40)}${condsAt(24, 350)}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, HX, "Tc_out", 0.5, 1, "°C"), o2 = tracked(q, s, tr, HX, "Thc", 0.5, 1, "°C")
    q('[data-vessel="E"]')?.classList.toggle("warn", o1 || o2)
    limited(q, "Tw", s.v.Tw, null, 45, `${fx(s.v.Tw, 1)} °C`)
    setText(q, '[data-value="Th_in"]', `${fx(s.v.Th_in, 1)} °C`)
    setText(q, '[data-value="hotc"]', tr(`${fx(s.cond.Fh, 1)} kg/s · measured`, `${fx(s.cond.Fh, 1)} kg/s · 可测`))
    setText(q, '[data-value="coldc"]', `${fx(s.cond.Tc_in, 1)} °C · ${fx(s.cond.Fc, 1)} kg/s`)
    setMv(q, "u0", s.v.u0, 0, 0.6, fx(s.v.u0, 3), 140); setMv(q, "u1", s.v.u1, 1, 40, `${fx(s.v.u1, 1)} kg/s`, 230)
  },
  flows: (s) => {
    const h = (s.cond.Fh ?? 10) / 12
    return { hin: h, byp: h * s.v.u0 * 2, tube: h * (1 - s.v.u0), mix: h, hout: h, cin: (s.cond.Fc ?? 12) / 14, cout: (s.cond.Fc ?? 12) / 14, cw: s.v.u1 / 30, ret: s.v.u1 / 30 }
  },
}

/* ---------- two-pass fired heater (furnace.py) ---------- */
const FU: Ind = { O2: [600, 30, 190], COT: [860, 230, 190], dT: [860, 316, 190] }
/* a pass: from the split node into the box, three runs across the radiant section, out through the right wall */
const coil = (y0: number, y1: number) => `M300 200V${y0}H600V${(y0 + y1) / 2}H400V${y1}H640`
const furnace: Screen = {
  cond: { lhv: [["fuel quality", "燃料热值"], pct], Ft: [["feed rate", "进料量"], f1(" kg/s")], Tin: [["inlet temperature", "入口温度"], f1(" °C")], kA: [["pass A heat transfer", "A 路传热"], pct], kB: [["pass B heat transfer", "B 路传热"], pct] },
  svg: (tr) => `${svgOpen(tr("Fired heater process screen, replay", "加热炉流程画面回放"), "fu-arrow")}
  <rect class="hmi-vessel-bg" x="380" y="70" width="240" height="250" rx="6"/>
  <rect class="hmi-vessel" data-vessel="box" x="380" y="70" width="240" height="250" rx="6"/><text class="hmi-sub" x="500" y="200" text-anchor="middle">${tr("radiant section", "辐射室")}</text>
  ${pipe("pa", coil(96, 168), "hmi-dr", "fu-arrow")}${pipe("pb", coil(232, 300), "hmi-dr", "fu-arrow")}
  <text class="hmi-tag" x="392" y="90">${tr("pass A", "A 路")}</text><text class="hmi-tag" x="392" y="226">${tr("pass B", "B 路")}</text>
  ${pipe("feed", "M40 200H298", "hmi-dr")}<circle class="hmi-node" cx="300" cy="200" r="3.5"/>${readout("Ft", 40, 188, tr("Feed Ft", "进料 Ft"), 170)}<text class="hmi-sub" data-value="feedc" x="40" y="222"></text>
  ${valve(300, 150, true)}${mv("u2", 40, 256, tr("s split to A", "s 分给 A 路"), 170)}
  ${pipe("fuel", "M220 342H470V322", "sh-u1", "fu-arrow")}${valve(300, 342)}${mv("u0", 40, 330, tr("f fuel", "f 燃料"), 170)}
  ${pipe("air", "M220 368H540V322", "sh-u2", "fu-arrow")}${valve(360, 368)}${mv("u1", 40, 362, tr("a air", "a 空气"), 170)}
  <path class="hmi-vessel" d="M455 318Q462 296 470 278Q478 296 485 318M525 318Q532 296 540 278Q548 296 555 318"/>
  ${pipe("stack", "M500 70V20", "hmi-dr", "fu-arrow")}${sensor("O2", 552, 44, "AT", "M500 44H537")}
  ${pipe("oa", "M640 168H800V198", "hmi-dr")}${pipe("ob", "M640 300H800V202", "hmi-dr")}<circle class="hmi-node" cx="800" cy="200" r="3.5"/>
  ${readout("TA", 650, 156, tr("TA ≤ 400", "TA ≤ 400"), 140)}${readout("TB", 650, 290, tr("TB ≤ 400", "TB ≤ 400"), 140)}
  ${pipe("cot", "M800 200H1056", "hmi-dr", "fu-arrow")}${sensor("COT", 850, 176, "TT", "M850 200V191")}
  ${inds(FU, { O2: tr("Excess O₂", "烟气 O₂"), COT: tr("COT coil outlet", "COT 出口温度"), dT: tr("TA − TB balance", "TA − TB 两路温差") })}
  ${clockBlock(1056, 36, "end")}${condsAt(1056, 98, "end")}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, FU, "COT", 1, 1, "°C"), o2 = tracked(q, s, tr, FU, "dT", 1, 1, "°C"), o3 = tracked(q, s, tr, FU, "O2", 0.2, 2, "%")
    const la = limited(q, "TA", s.v.TA, null, 400, `${fx(s.v.TA, 1)} °C`), lb = limited(q, "TB", s.v.TB, null, 400, `${fx(s.v.TB, 1)} °C`)
    q('[data-vessel="box"]')?.classList.toggle("warn", o1 || o2 || o3 || la || lb)
    setText(q, '[data-value="Ft"]', `${fx(s.v.Ft, 1)} kg/s`)
    setText(q, '[data-value="feedc"]', tr(`inlet ${fx(s.cond.Tin, 0)} °C · measured`, `入口 ${fx(s.cond.Tin, 0)} °C · 可测`))
    setMv(q, "u0", s.v.u0, 0.15, 0.8, `${fx(s.v.u0, 3)} kg/s`, 170); setMv(q, "u1", s.v.u1, 3, 18, `${fx(s.v.u1, 2)} kg/s`, 170); setMv(q, "u2", s.v.u2, 0.3, 0.7, fx(s.v.u2, 3), 170)
  },
  flows: (s) => { const f = s.v.Ft / 55; return { feed: f, pa: f * s.v.u2 * 2, pb: f * (1 - s.v.u2) * 2, oa: f * s.v.u2 * 2, ob: f * (1 - s.v.u2) * 2, cot: f, fuel: s.v.u0 / 0.6, air: s.v.u1 / 14, stack: s.v.u1 / 14 } },
}

/* ---------- gas transmission line (gasnet.py) ---------- */
const GN: Ind = { q_in: [130, 66, 180], P2: [860, 66, 180] }
const gasnet: Screen = {
  cond: { P0: [["source pressure", "气源压力"], f1(" bar")], qd: [["demand", "用户负荷"], f1(" kg/s")], kf: [["line friction", "管线摩阻"], pct], Plp: [["low-pressure system", "低压管网压力"], f1(" bar")], cv1: [["supply valve capacity", "供气阀流通能力"], pct] },
  svg: (tr) => `${svgOpen(tr("Gas pipeline process screen, replay", "输气管线流程画面回放"), "gn-arrow")}
  <rect class="hmi-vessel-bg" x="24" y="176" width="80" height="48" rx="8"/><rect class="hmi-vessel" x="24" y="176" width="80" height="48" rx="8"/>
  <text class="hmi-tag" x="64" y="196" text-anchor="middle">${tr("Source", "气源")}</text><text class="hmi-value" data-value="P0" x="64" y="214" text-anchor="middle">—</text>
  ${pipe("sup", "M104 200H298", "sh-u1", "gn-arrow")}${valve(200, 200)}${sensor("q_in", 150, 160, "FT", "M150 200V175")}${mv("u0", 130, 250, tr("u1 supply valve", "u1 供气阀"), 150)}
  <circle class="hmi-node" cx="300" cy="200" r="3.5"/>${sensor("P1", 330, 160, "PT", "M330 200V175")}${readout("P1", 300, 250, tr("P1 inlet ≤ 66", "P1 入口 ≤ 66"), 150)}<text class="hmi-sub" x="300" y="268">bar</text>
  <rect class="hmi-vessel-bg" x="360" y="186" width="380" height="28" rx="14"/>
  <clipPath id="gn-clip"><rect x="362" y="188" width="376" height="24" rx="12"/></clipPath><g clip-path="url(#gn-clip)"><rect class="hmi-liquid" data-liquid="pack" x="360" y="214" width="380" height="28"/></g>
  <rect class="hmi-vessel" x="360" y="186" width="380" height="28" rx="14"/>${pipe("line", "M300 200H798", "hmi-dr")}
  <text class="hmi-sub" x="550" y="176" text-anchor="middle">${tr("50 km line · shading = line pack", "50 km 管线 · 阴影 = 管存")}</text>
  <circle class="hmi-node" cx="800" cy="200" r="3.5"/>${sensor("P2", 770, 160, "PT", "M770 200V175")}<text class="hmi-tag" x="790" y="240" text-anchor="end">${tr("City gate", "城门站")}</text>
  ${pipe("dem", "M800 200V320", "hmi-dr", "gn-arrow")}${readout("qd", 812, 312, tr("Consumers qd", "用户 qd"), 200)}<text class="hmi-sub" x="812" y="330">${tr("measured", "可测")}</text>
  ${pipe("ld", "M800 200H1056", "sh-u2", "gn-arrow")}${valve(900, 200)}${mv("u1", 880, 250, tr("u2 let-down valve", "u2 泄放阀"), 170)}
  <text class="hmi-sub" data-value="lp" x="1056" y="186" text-anchor="end"></text>
  ${inds(GN, { q_in: tr("q_in supply flow", "q_in 供气量"), P2: tr("P2 city gate", "P2 城门站压力") })}
  ${clockBlock(540, 64, "middle")}${condsAt(24, 350)}
  </svg>`,
  update: (q, s, tr) => {
    tracked(q, s, tr, GN, "q_in", 0.5, 1, "kg/s"); tracked(q, s, tr, GN, "P2", 0.1, 2, "bar")
    limited(q, "P1", s.v.P1, null, 66, `${fx(s.v.P1, 2)}`)
    const pk = Math.min(Math.max(((s.v.P1 + s.v.P2) / 2 - 40) / 30, 0), 1)
    q('[data-liquid="pack"]')?.setAttribute("y", (214 - pk * 28).toFixed(1))
    setText(q, '[data-value="P0"]', `${fx(s.cond.P0, 1)} bar`); setText(q, '[data-value="qd"]', `${fx(s.v.qd, 1)} kg/s`)
    setText(q, '[data-value="lp"]', tr(`to low pressure, ${fx(s.cond.Plp, 1)} bar`, `去低压管网，${fx(s.cond.Plp, 1)} bar`))
    setMv(q, "u0", s.v.u0, 0, 1, fx(s.v.u0, 3), 150); setMv(q, "u1", s.v.u1, 0, 1, fx(s.v.u1, 3), 170)
  },
  flows: (s) => ({ sup: s.v.q_in / 45, line: s.v.q_in / 45, dem: s.v.qd / 40, ld: Math.max(0, s.v.q_in - s.v.qd) / 20 }),
}

/* ---------- procedural random plants (proc_batch.py): one block, the plant's own MVs and outputs ---------- */
type Row = { k: string; y: number; kind: "track" | "con" | "dist" }
function procRows(d: Layout) {
  const rows: Row[] = [], hgt = (c: Layout["cvs"][number]) => (c.tracked ? 56 : 40)
  const cvs = [...d.cvs].sort((a, b) => (a.tracked === b.tracked ? 0 : a.tracked ? -1 : 1))
  const total = cvs.reduce((a, c) => a + hgt(c), 0)
  let y = 40 + Math.max(0, (310 - total) / 2)
  for (const c of cvs) { rows.push({ k: c.name || c.key, y: y + 12, kind: c.tracked ? "track" : c.lower != null || c.upper != null ? "con" : "dist" }); y += hgt(c) }
  const mys = d.mvs.map((_, j) => 70 + ((j + 0.5) * 260) / d.mvs.length)
  return { rows, mys }
}
const limits = (c: { lower?: number | null; upper?: number | null; dec: number }, tr: Tr) =>
  c.lower != null && c.upper != null ? tr(`limits ${fx(c.lower, c.dec)} to ${fx(c.upper, c.dec)}`, `限值 ${fx(c.lower, c.dec)} 至 ${fx(c.upper, c.dec)}`)
  : c.lower != null ? tr(`at least ${fx(c.lower, c.dec)}`, `不低于 ${fx(c.lower, c.dec)}`) : c.upper != null ? tr(`at most ${fx(c.upper, c.dec)}`, `不高于 ${fx(c.upper, c.dec)}`) : tr("measured disturbance", "实测扰动")
/* proc, its second generation proc2 (proc2_batch.py: sticky valves, load-dependent dynamics, extra lags, unstable rows,
   named in the MV and CV labels) and mimo (mimo_batch.py: proc2 shifted to APC-scale plants, 2-5 coupled loops, long dead
   times) share one screen */
type ProcKind = "proc" | "proc2" | "mimo"
const procName = (k: ProcKind, tr: Tr) => (k === "mimo" ? tr("APC-scale random plant", "APC 规模随机装置") : k === "proc2" ? tr("Random plant, 2nd gen", "第二代随机装置") : tr("Random plant", "随机装置"))
const procScreen = (kind: ProcKind): Screen => ({
  cond: {},
  conds: (c, tr) => [`${procName(kind, tr)} · ${tr(`${c.dt} s step`, `每步 ${c.dt} 秒`)}`,
    c.nchg ? tr(`plant changes so far: ${c.nchg}`, `已发生装置变化 ${c.nchg} 次`) : tr("plant as at the start", "装置与起点相同")],
  svg: (tr, d) => {
    const { rows, mys } = procRows(d), lab = (c: { label: string; label_zh?: string }) => esc(tr(c.label, c.label_zh || c.label))
    const mvs = d.mvs.map((m, j) => `${pipe(`m${j}`, `M40 ${mys[j] + 22}H398`, MVC[j % MVC.length], "pr-arrow")}${valve(300, mys[j] + 22)}${mv(m.key, 40, mys[j], lab(m), 200)}`).join("")
    const cvs = rows.map((r) => {
      const c = d.cvs.find((x) => (x.name || x.key) === r.k)!
      const line = r.kind === "dist" ? pipe(`c-${r.k}`, `M750 ${r.y + 4}H622`, "hmi-dr", "pr-arrow") : pipe(`c-${r.k}`, `M622 ${r.y + 4}H740`, "hmi-dr", "pr-arrow")
      return line + (r.kind === "track" ? indicator(r.k, 760, r.y, 280, lab(c)) : readout(r.k, 760, r.y, lab(c), 280, limits(c, tr)))
    }).join("")
    const st = d.start || {}
    return `${svgOpen(tr("Random plant process screen, replay", "随机装置流程画面回放"), "pr-arrow")}
  <rect class="hmi-vessel-bg" x="400" y="70" width="220" height="260" rx="12"/><rect class="hmi-vessel" data-vessel="P" x="400" y="70" width="220" height="260" rx="12"/>
  <text class="hmi-tag" x="510" y="170" text-anchor="middle">${procName(kind, tr)}</text>
  <text class="hmi-sub" x="510" y="192" text-anchor="middle">${tr(`${st.nm} MVs → ${st.nt} tracked CVs`, `${st.nm} 个 MV → ${st.nt} 个跟踪 CV`)}</text>
  <text class="hmi-sub" x="510" y="212" text-anchor="middle">${tr("gains, lags, dead times hidden", "增益、惯性、纯滞后不公开")}</text>
  ${kind !== "proc" ? `<text class="hmi-sub" x="510" y="232" text-anchor="middle">${tr("friction, load and extra lags too", "粘滞、负荷和附加滞后也不公开")}</text>` : ""}
  ${mvs}${cvs}
  ${clockBlock(24, 40)}${condsAt(24, 350)}
  </svg>`
  },
  update: (q, s, tr, d) => {
    let bad = false
    const ys = Object.fromEntries(procRows(d).rows.map((r) => [r.k, r.y]))
    for (const c of d.cvs) {
      const k = c.name || c.key, v = s.v[k]
      if (c.tracked) {
        const r = s.ref[k], b = c.band || 0
        setIndicator(q, k, 760, 280, ys[k], v, r, b, fx(v, c.dec), tr(`target ${fx(r, c.dec)} ± ${fx(b, c.dec)}`, `目标 ${fx(r, c.dec)} ± ${fx(b, c.dec)}`))
        bad = bad || Math.abs(v - r) > b
      } else bad = limited(q, k, v, c.lower ?? null, c.upper ?? null, fx(v, c.dec)) || bad
    }
    q('[data-vessel="P"]')?.classList.toggle("warn", bad)
    for (const m of d.mvs) setMv(q, m.key, s.v[m.key], m.lo ?? 0, m.hi ?? 1, fx(s.v[m.key], m.dec), 200)
  },
  flows: (s, d) => {
    const out: Record<string, number> = {}
    d.mvs.forEach((m, j) => { out[`m${j}`] = 0.2 + 0.8 * ((s.v[m.key] - (m.lo ?? 0)) / ((m.hi ?? 1) - (m.lo ?? 0))) })
    for (const c of d.cvs) out[`c-${c.name || c.key}`] = 0.4
    return out
  },
})
const proc = procScreen("proc"), proc2 = procScreen("proc2"), mimo = procScreen("mimo")

/* ---------- random flowsheets (netplant_batch.py): drawn from each scenario's own structure (the file's `net`) ----------
   Units in chain order on one line, recycle loops below (inner loop higher), utilities, side feed, purge and flash vapour
   above; a sensor for every published CV on the stream it measures; MV pipes in the MV colours; the CV column on the right
   and the MV readouts along the bottom, as on the random-plant screen. */
const NET_Y = 176, NET_HW: Record<string, number> = { mixer: 18, cstr: 24, hx: 20, flash: 17, splitter: 12, pipe: 24 }
const NET_LOOP = [248, 268], NET_SENS = [226, 128]
const NET_UNIT: Record<string, string> = { T: "K", A: "kmol/m³", P: "kmol/min", Y: "", Qs: "m³/min", cA0: "kmol/m³", T0: "K", F0: "m³/min", Fs: "m³/min", Tu: "K", rho: "" }
const NET_SENSOR: Record<string, string> = { T: "TT", A: "AT", P: "B", Y: "Y", Qs: "FT", cA0: "AT", T0: "TT" }
function netGeo(n: NetFlow) {
  const L = n.units.length, step = L > 1 ? Math.min(84, 520 / (L - 1)) : 0, x0 = 130 + (520 - (L - 1) * step) / 2
  const cx = n.units.map((_, k) => x0 + k * step), hw = n.units.map((u) => NET_HW[u.kind])
  const top = n.units.map((u, k) => (u.kind === "flash" ? NET_Y - 32 : u.kind === "cstr" ? NET_Y - 26 : u.kind === "pipe" ? NET_Y - 7 : NET_Y - hw[k]))
  const bot = n.units.map((u, k) => (u.kind === "flash" ? NET_Y + 32 : u.kind === "cstr" ? NET_Y + 26 : u.kind === "pipe" ? NET_Y + 7 : NET_Y + hw[k]))
  /* a sensor per CV on the stream it measures: a unit's outlet between it and the next unit (below the line, a second one
     above), the feed left of the first unit, the product right of the last, a recycle flow on its loop, a purge on its line */
  const used: Record<number, number> = {}, at: Record<string, { x: number; y: number; tap: boolean }> = {}
  const slot = (x: number) => { const i = (used[x] = (used[x] ?? -1) + 1); return NET_SENS[Math.min(i, 1)] }
  const half = Math.max(step / 2, 30)
  for (const c of n.cvs) {
    let x: number
    if (c.y === "Qs" && n.loops.some(([s]) => s === c.unit)) { at[c.name] = { x: cx[c.unit], y: NET_SENS[0], tap: false }; continue }
    if (c.y === "Qs") { at[c.name] = { x: cx[c.unit] + 13, y: 124, tap: false }; continue }
    if (c.unit >= 0) x = cx[c.unit] + half
    else if (c.y === "cA0" || c.y === "T0") x = cx[0] - hw[0] - 24
    else x = Math.max(cx[L - 1] + half + 37, 724)
    at[c.name] = { x, y: slot(x), tap: true }
  }
  return { L, step, cx, hw, top, bot, at }
}
/* the screen tag of an MV (u1 F0 feed), from the handle it moves */
const netMvTag = (n: NetFlow, j: number, tr: Tr) => {
  const m = n.mvs[j], u = m.unit >= 0 ? n.units[m.unit] : null, t = u?.tag || ""
  const what = m.h === "F0" ? tr("F0 feed", "F0 进料") : m.h === "Fs" ? tr("Fs side feed", "Fs 侧线进料") : m.h === "rho" ? tr(`${t} split`, `${t} 分流比`)
    : u?.kind === "cstr" ? tr(`${t} jacket T`, `${t} 夹套温度`) : u?.kind === "hx" ? tr(`${t} utility T`, `${t} 介质温度`) : tr(`${t} heating T`, `${t} 加热温度`)
  return `u${j + 1} ${what}`
}
function netUnit(u: NetFlow["units"][number], k: number, x: number) {
  const id = `n${k}`, y = NET_Y
  switch (u.kind) {
    case "mixer": return `<circle class="hmi-vessel-bg" cx="${x}" cy="${y}" r="18"/><circle class="hmi-vessel" data-vessel="${id}" cx="${x}" cy="${y}" r="18"/>`
    case "hx": return `<circle class="hmi-vessel-bg" cx="${x}" cy="${y}" r="20"/><circle class="hmi-vessel" data-vessel="${id}" cx="${x}" cy="${y}" r="20"/><path class="tep-coil" d="M${x - 16} ${y}H${x - 7}L${x - 2} ${y - 9} ${x + 3} ${y + 9} ${x + 7} ${y}H${x + 16}"/>`
    case "splitter": return `<path class="hmi-vessel-bg" d="M${x} ${y - 12}L${x + 12} ${y}L${x} ${y + 12}L${x - 12} ${y}Z"/><path class="hmi-vessel" data-vessel="${id}" d="M${x} ${y - 12}L${x + 12} ${y}L${x} ${y + 12}L${x - 12} ${y}Z"/>`
    case "pipe": return `<rect class="hmi-vessel-bg" x="${x - 24}" y="${y - 7}" width="48" height="14" rx="7"/><rect class="hmi-vessel" data-vessel="${id}" x="${x - 24}" y="${y - 7}" width="48" height="14" rx="7"/>`
    default: {   // cstr (jacketed, stirred) and flash drum: a vessel with liquid
      const [w, h, lq] = u.kind === "cstr" ? [48, 52, 0.35] : [34, 64, 0.55], x0 = x - w / 2, y0 = y - h / 2
      return `<rect class="hmi-vessel-bg" x="${x0}" y="${y0}" width="${w}" height="${h}" rx="${u.kind === "cstr" ? 12 : 10}"/>
  <clipPath id="nt-clip-${k}"><rect x="${x0 + 2}" y="${y0 + 2}" width="${w - 4}" height="${h - 4}" rx="9"/></clipPath>
  <g clip-path="url(#nt-clip-${k})"><rect class="hmi-liquid" x="${x0}" y="${y0 + h * lq}" width="${w}" height="${h}"/><line class="hmi-surface" x1="${x0}" x2="${x0 + w}" y1="${y0 + h * lq}" y2="${y0 + h * lq}"/></g>
  ${u.kind === "cstr" ? `<path class="hmi-vessel" d="M${x0 - 5} ${y0 + h * 0.4}V${y0 + h - 6}Q${x0 - 5} ${y0 + h + 5} ${x0 + 8} ${y0 + h + 5}H${x0 + w - 8}Q${x0 + w + 5} ${y0 + h + 5} ${x0 + w + 5} ${y0 + h - 6}V${y0 + h * 0.4}"/>` : ""}
  <rect class="hmi-vessel" data-vessel="${id}" x="${x0}" y="${y0}" width="${w}" height="${h}" rx="${u.kind === "cstr" ? 12 : 10}"/>`
    }
  }
}
const net: Screen = {
  cond: {},
  conds: (c, tr) => [tr(`Random flowsheet · ${c.dt} s step · ${c.nchg ? `plant changes so far: ${c.nchg}` : "plant as at the start"}`, `随机流程图 · 每步 ${c.dt} 秒 · ${c.nchg ? `已发生工况变化 ${c.nchg} 次` : "工况与起点相同"}`),
    [tr(`catalyst ${pct(c.act)}`, `催化剂活性 ${pct(c.act)}`), ...(c.kapf != null ? [tr(`flash ${pct(c.kapf)}`, `闪蒸效率 ${pct(c.kapf)}`)] : []),
      tr(`feed A ${fx(c.cA0, 2)} kmol/m³, ${fx(c.T0, 1)} K`, `进料 A ${fx(c.cA0, 2)} kmol/m³，${fx(c.T0, 1)} K`)].join(" · ")],
  svg: (tr, d) => {
    const n = d.net!, g = netGeo(n), { L, cx, hw, top, bot } = g, Y = NET_Y
    const mvOf = (h: string, k = -1) => n.mvs.findIndex((m) => m.h === h && (k < 0 || m.unit === k))
    const col = (j: number) => (j >= 0 ? MVC[j % MVC.length] : "hmi-dr")
    const lab = (c: { label: string; label_zh?: string }) => esc(tr(c.label, c.label_zh || c.label))
    let s = ""
    /* feed and product */
    const f0 = mvOf("F0"), xin = cx[0] - hw[0] - 2, xout = 748
    s += pipe("feed", `M24 ${Y}H${xin}`, col(f0), "nt-arrow") + valve(54, Y) + `<text class="hmi-sub" x="24" y="${Y - 14}">${f0 >= 0 ? `u${f0 + 1}` : "F0"}</text>`
    s += pipe("prod", `M${cx[L - 1] + hw[L - 1]} ${Y}H${xout}`, "hmi-dr", "nt-arrow") + `<text class="hmi-sub" x="${xout}" y="${Y - 10}" text-anchor="end">${tr("product", "产品")}</text>`
    /* the chain */
    for (let k = 0; k < L - 1; k++) s += pipe(`c${k}`, `M${cx[k] + hw[k]} ${Y}H${cx[k + 1] - hw[k + 1] - 2}`, "hmi-dr", "nt-arrow")
    /* recycle loops: down from the splitter, back under the chain, up into the unit they return to */
    n.loops.forEach(([a, b], i) => {
      const j = mvOf("rho", a), yl = NET_LOOP[Math.min(i, 1)]
      s += pipe(`l${i}`, `M${cx[a]} ${bot[a]}V${yl}H${cx[b]}V${bot[b] + 2}`, col(j), "nt-arrow")
      if (j >= 0) s += `<text class="hmi-sub" x="${cx[a] + 6}" y="${yl - 6}">u${j + 1}</text>`
    })
    /* above the line: utilities (Tu), side feed, purges, flash vapour */
    n.units.forEach((u, k) => {
      const x = cx[k]
      if (u.kind === "cstr" || u.kind === "hx" || u.kind === "flash") {
        const j = mvOf("Tu", k), xu = u.kind === "flash" ? x - 11 : x + 13
        s += j >= 0 ? pipe(`t${k}`, `M${xu} 98V${top[k] - 1}`, col(j), "nt-arrow") + `<text class="hmi-sub" x="${xu}" y="92" text-anchor="middle">u${j + 1}</text>`
          : `<path class="tep-util" d="M${xu} 104V${top[k] - 1}"/>`
      }
      if (u.kind === "flash") s += pipe(`v${k}`, `M${x + 11} ${top[k]}V98`, "hmi-dr", "nt-arrow") + `<text class="hmi-sub" x="${x + 17}" y="110">${tr("vapour", "汽相")}</text>`
      if (n.side === k) {
        const j = mvOf("Fs")
        s += pipe("side", `M${x + 12} 98V${top[k] - 1}`, col(j), "nt-arrow") + valve(x + 12, 122, true) + `<text class="hmi-sub" x="${x + 12}" y="92" text-anchor="middle">${j >= 0 ? `u${j + 1}` : "Fs"}</text>`
      }
      if (n.purge.includes(k)) {
        const j = mvOf("rho", k)
        s += pipe(`p${k}`, `M${x + 13} ${top[k] + 6}V98`, col(j), "nt-arrow") + `<text class="hmi-sub" x="${x + 13}" y="92" text-anchor="middle">${j >= 0 ? `u${j + 1}` : tr("purge", "排放")}</text>`
      }
      s += netUnit(u, k, x) + `<text class="hmi-tag" x="${x}" y="${top[k] - 8}" text-anchor="middle">${u.tag}</text>`
    })
    /* sensors on the streams they measure */
    for (const c of n.cvs) {
      const { x, y, tap } = g.at[c.name]
      s += sensor(c.name, x, y, NET_SENSOR[c.y], tap ? `M${x} ${Y}V${y > Y ? y - 15 : y + 15}` : undefined)
    }
    /* CV column and MV readouts */
    const { rows } = procRows(d)
    s += rows.map((r) => {
      const c = d.cvs.find((x) => (x.name || x.key) === r.k)!
      return r.kind === "track" ? indicator(r.k, 760, r.y, 280, lab(c)) : readout(r.k, 760, r.y, lab(c), 280, limits(c, tr))
    }).join("")
    const per = Math.min(d.mvs.length, 3), w = Math.min(200, 680 / per - 30)
    s += d.mvs.map((m, j) => {
      const x = 40 + (j % 3) * (w + 30), y = 292 + Math.floor(j / 3) * 34
      return `<path class="hmi-flow ${MVC[j % MVC.length]}" d="M${x} ${y - 4}h14"/><text class="hmi-tag" x="${x + 20}" y="${y}">${esc(netMvTag(n, j, tr))}</text><text class="hmi-value" data-value="${m.key}" x="${x + w}" y="${y}" text-anchor="end">—</text>${bar(m.key, x + 20, y + 8, w - 20)}`
    }).join("")
    const nl = n.loops.length
    return `${svgOpen(tr("Random flowsheet process screen, replay", "随机流程图流程画面回放"), "nt-arrow")}
  <text class="hmi-note" x="440" y="28" text-anchor="middle">${tr(`A → B → C, 2A → C · ${L} units, ${nl} recycle loop${nl === 1 ? "" : "s"}`, `A → B → C，2A → C · ${L} 个单元、${nl} 个循环回路`)}</text>
  ${s}
  ${clockBlock(24, 40)}${condsAt(24, 352)}
  </svg>`
  },
  update: (q, s, tr, d) => {
    const n = d.net!, ys = Object.fromEntries(procRows(d).rows.map((r) => [r.k, r.y])), bad: Record<number, boolean> = {}
    for (const c of d.cvs) {
      const k = c.name || c.key, v = s.v[k], f = n.cvs.find((x) => x.name === k), u = f ? NET_UNIT[f.y] : "", t = (x: number) => `${fx(x, Math.min(c.dec, 3))}${u ? ` ${u}` : ""}`
      let out: boolean
      if (c.tracked) {
        const r = s.ref[k], b = c.band || 0
        setIndicator(q, k, 760, 280, ys[k], v, r, b, t(v), tr(`target ${fx(r, Math.min(c.dec, 3))} ± ${+b.toPrecision(3)}`, `目标 ${fx(r, Math.min(c.dec, 3))} ± ${+b.toPrecision(3)}`))
        out = Math.abs(v - r) > b
        q(`[data-at="${k}"]`)?.classList.toggle("warn", out)
      } else out = limited(q, k, v, c.lower ?? null, c.upper ?? null, t(v))
      if (f && f.unit >= 0 && out) bad[f.unit] = true
    }
    n.units.forEach((_, k) => q(`[data-vessel="n${k}"]`)?.classList.toggle("warn", !!bad[k]))
    d.mvs.forEach((m, j) => {
      const per = Math.min(d.mvs.length, 3), w = Math.min(200, 680 / per - 30), u = NET_UNIT[n.mvs[j].h]
      setMv(q, m.key, s.v[m.key], m.lo ?? 0, m.hi ?? 1, `${fx(s.v[m.key], Math.min(m.dec, 3))}${u ? ` ${u}` : ""}`, w - 20)
    })
  },
  flows: (s, d) => {
    const n = d.net!, out: Record<string, number> = {}, frac = (j: number) => { const m = d.mvs[j]; return 0.2 + 0.8 * ((s.v[m.key] - (m.lo ?? 0)) / ((m.hi ?? 1) - (m.lo ?? 0))) }
    const f0 = n.mvs.findIndex((m) => m.h === "F0"), base = f0 >= 0 ? frac(f0) : 0.5
    for (const k of ["feed", "prod", "side", ...n.units.map((_, k) => `c${k}`)]) out[k] = base
    n.loops.forEach((_, i) => { out[`l${i}`] = 0.4 })
    n.units.forEach((_, k) => { out[`v${k}`] = 0.4; out[`p${k}`] = 0.4 })
    n.mvs.forEach((m, j) => {
      if (m.h === "Tu") out[`t${m.unit}`] = frac(j)
      else if (m.h === "Fs") out.side = frac(j)
      else if (m.h === "rho") { const i = n.loops.findIndex(([a]) => a === m.unit); if (i >= 0) out[`l${i}`] = frac(j); else out[`p${m.unit}`] = frac(j) }
    })
    return out
  },
}

/* ---------- held-out families (docs/heldout.md): never trained, tested zero-shot; topology and tags from experiments/scenario_ppo/heldout/*.py ---------- */

/* ---------- drum boiler-turbine unit (drumboiler.py) ---------- */
const BO: Ind = { E: [880, 112, 170], P: [880, 202, 170] }
const BO_DRUM = { x: 380, y: 56, w: 300, h: 96 }
const boY = (L: number) => BO_DRUM.y + BO_DRUM.h / 2 - (Math.min(Math.max(L, -0.2), 0.2) / 0.2) * 40
const boiler: Screen = {
  cond: { fq: [["fuel heating value", "燃料热值"], pct], fw: [["feedwater subcooling", "给水欠热"], pct], eta: [["turbine efficiency", "汽轮机效率"], pct], kv: [["steam valve coefficient", "汽门系数"], pct] },
  svg: (tr) => {
    const { x, y, w, h } = BO_DRUM
    return `${svgOpen(tr("Drum boiler process screen, replay", "汽包锅炉流程画面回放"), "bo-arrow")}
  <text class="hmi-note" x="530" y="186" text-anchor="middle">${tr("swell: level first rises, then falls", "虚假水位：开大汽门，水位先涨后落")}</text>
  <rect class="hmi-vessel-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="44"/>
  <clipPath id="bo-clip"><rect x="${x + 2}" y="${y + 2}" width="${w - 4}" height="${h - 4}" rx="42"/></clipPath>
  <g clip-path="url(#bo-clip)"><rect class="hmi-liquid" data-liquid="L" x="${x}" y="${y + h}" width="${w}" height="${h}"/><line class="hmi-surface" data-surface="L" x1="${x}" x2="${x + w}" y1="${y + h}" y2="${y + h}"/>
  <rect class="qt-band" data-band="L" x="${x}" y="0" width="${w}" height="0"/><line class="qt-sp" data-sp="L" x1="${x}" x2="${x + w}" y1="0" y2="0"/></g>
  ${[-0.1, 0.1].map((l) => `<path class="hmi-limit" d="M${x + 30} ${boY(l).toFixed(1)}h12M${x + w - 30} ${boY(l).toFixed(1)}h-12"/>`).join("")}
  <rect class="hmi-vessel" data-vessel="L" x="${x}" y="${y}" width="${w}" height="${h}" rx="44"/>
  <text class="hmi-tag" x="${x + 40}" y="${y + 24}">${tr("Drum", "汽包")}</text><text class="hmi-sub" x="${x + w - 40}" y="${y + 24}" text-anchor="end">${tr("level L, limits ±0.1 m", "水位 L，限值 ±0.1 m")}</text>
  <text class="hmi-value" data-value="L" x="${x + w / 2}" y="${y + 44}" text-anchor="middle">—</text><text class="hmi-sub" data-sub="L" x="${x + w / 2}" y="${y + 62}" text-anchor="middle"></text>
  ${pipe("dc", "M420 146V208", "hmi-dr", "bo-arrow")}${pipe("rise", "M640 208V148", "hmi-dr", "bo-arrow")}
  <rect class="hmi-vessel-bg" x="400" y="210" width="260" height="110" rx="8"/><path class="tep-tray" d="M440 220V310M480 220V310M520 220V310M560 220V310M600 220V310M640 220V310"/>
  <path class="hmi-vessel" d="M455 316Q462 294 470 276Q478 294 485 316M575 316Q582 294 590 276Q598 294 605 316"/>
  <rect class="hmi-vessel" x="400" y="210" width="260" height="110" rx="8"/><text class="hmi-sub" x="530" y="340" text-anchor="middle">${tr("furnace · riser tubes", "炉膛 · 上升管")}</text>
  ${pipe("fuel", "M40 300H398", "sh-u1", "bo-arrow")}${valve(250, 300)}${mv("u0", 40, 272, tr("u1 fuel valve", "u1 燃料阀"), 170)}
  ${pipe("fw", "M40 120H378", "sh-u3", "bo-arrow")}${valve(250, 120)}${mv("u2", 40, 146, tr("u3 feedwater valve", "u3 给水阀"), 170)}
  ${pipe("steam", "M600 58V30H798", "sh-u2", "bo-arrow")}${valve(720, 30)}${mv("u1", 690, 70, tr("u2 steam", "u2 汽门"), 108)}
  ${sensor("P", 520, 34, "PT", "M520 58V49")}
  <path class="hmi-vessel" d="M800 16L856 4V58L800 46Z"/><text class="hmi-sub" x="828" y="76" text-anchor="middle">${tr("turbine", "汽轮机")}</text>
  <path class="tep-util" d="M856 31H898"/><circle class="hmi-vessel-bg" cx="920" cy="31" r="22"/><circle class="hmi-vessel" data-at="E" cx="920" cy="31" r="22"/><text class="hmi-tag" x="920" y="36" text-anchor="middle">G</text>
  ${readout("qs", 690, 122, tr("qs steam", "qs 蒸汽流量"), 150, tr("monitored", "只监视"))}
  ${inds(BO, { E: tr("E power", "E 电功率"), P: tr("P pressure", "P 汽包压力") })}
  ${clockBlock(24, 40)}${condsAt(24, 350)}
  </svg>`
  },
  update: (q, s, tr) => {
    const L = s.v.L, r = s.ref.L, yL = boY(L), viol = Math.abs(L) > 0.1, out = Math.abs(L - r) > 0.02
    q('[data-liquid="L"]')?.setAttribute("y", yL.toFixed(2)); const sf = q('[data-surface="L"]'); sf?.setAttribute("y1", yL.toFixed(2)); sf?.setAttribute("y2", yL.toFixed(2))
    const b1 = boY(r + 0.02), b2 = boY(r - 0.02), band = q('[data-band="L"]'), sp = q('[data-sp="L"]')
    band?.setAttribute("y", b1.toFixed(1)); band?.setAttribute("height", (b2 - b1).toFixed(1)); sp?.setAttribute("y1", boY(r).toFixed(1)); sp?.setAttribute("y2", boY(r).toFixed(1))
    q('[data-vessel="L"]')?.classList.toggle("warn", viol || out)
    setText(q, '[data-value="L"]', `${fx(L, 3)} m`, viol || out); setText(q, '[data-sub="L"]', tr(`target ${fx(r, 3)} ± 0.02`, `目标 ${fx(r, 3)} ± 0.02`))
    tracked(q, s, tr, BO, "E", 1, 1, "MW"); tracked(q, s, tr, BO, "P", 1, 1, "kg/cm²")
    setText(q, '[data-value="qs"]', `${fx(s.v.qs, 1)} kg/s`)
    setMv(q, "u0", s.v.u0, 0, 1, fx(s.v.u0, 3), 170); setMv(q, "u1", s.v.u1, 0, 1, fx(s.v.u1, 3), 108); setMv(q, "u2", s.v.u2, 0, 1, fx(s.v.u2, 3), 170)
  },
  flows: (s) => ({ fuel: s.v.u0 / 0.4, fw: s.v.u2 / 0.5, steam: s.v.qs / 90, dc: 0.5, rise: 0.5 }),
}

/* ---------- Williams-Otto reactor with separation and recycle (williamsotto.py) ---------- */
const WO: Ind = { P: [880, 130, 170], XG: [880, 210, 170], R: [880, 290, 170] }
const wo: Screen = {
  cond: { FA: [["feed A", "A 进料"], f3(" kg/s")], a: [["catalyst activity", "催化剂活性"], pct], s3: [["by-product selectivity", "副反应选择性"], pct], tauR: [["recycle drum time constant", "循环罐时间常数"], (v: number) => `${fx(v, 0)} s`] },
  svg: (tr) => `${svgOpen(tr("Williams–Otto plant process screen, replay", "Williams–Otto 装置流程画面回放"), "wo-arrow")}
  <text class="hmi-note" x="480" y="28" text-anchor="middle">A + B → C · B + C → P + E · C + P → G</text>
  ${pipe("fb", "M40 64H390V118", "sh-u1", "wo-arrow")}${valve(200, 64)}${mv("u0", 40, 40, tr("FB feed B", "FB B 进料"), 150)}
  ${pipe("fa", "M40 100H350V118", "hmi-dr", "wo-arrow")}<text class="hmi-tag" x="40" y="92">${tr("Feed A", "A 进料")}</text><text class="hmi-sub" data-value="fa" x="190" y="92" text-anchor="end"></text>
  ${cstr("R", 300, 120, 150, 170, true, tr("Reactor", "反应器"))}
  <circle class="hmi-vessel-bg" cx="240" cy="220" r="18"/><circle class="hmi-vessel" cx="240" cy="220" r="18"/><text class="hmi-sub" x="240" y="224" text-anchor="middle">TC</text>
  <path class="tep-util" d="M258 220H298"/><path class="tep-util sh-u2" d="M200 220H222"/>${mv("u1", 40, 196, tr("TR reactor T", "TR 反应温度"), 150)}
  <text class="hmi-sub" x="40" y="246">${tr("base layer holds it", "基础层保持")}</text>
  ${pipe("out", "M450 250H558", "hmi-dr", "wo-arrow")}${readout("F", 468, 238, "F ≤ 16", 88)}<text class="hmi-sub" x="468" y="270">${tr("total, kg/s", "总通量 kg/s")}</text>
  <rect class="hmi-vessel-bg" x="560" y="110" width="90" height="190" rx="18"/><path class="tep-tray" d="M570 150H640M570 190H640M570 230H640M570 270H640" stroke-dasharray="4 4"/>
  <rect class="hmi-vessel" data-vessel="S" x="560" y="110" width="90" height="190" rx="18"/><text class="hmi-tag" x="605" y="100" text-anchor="middle">${tr("Separation", "分离")}</text>
  ${pipe("prod", "M650 160H860", "hmi-dr", "wo-arrow")}${sensor("P", 760, 136, "AT", "M760 160V151")}<text class="hmi-sub" x="860" y="180" text-anchor="end">${tr("P and G out", "P、G 出料")}</text>
  ${pipe("rec", "M605 300V350H380V306", "hmi-dr", "wo-arrow")}${sensor("R", 500, 326, "FT", "M500 350V341")}
  ${pipe("purge", "M605 350H860", "sh-u3", "wo-arrow")}${valve(720, 350)}${mv("u2", 680, 322, tr("η purge share", "η 排放比"), 170)}
  ${inds(WO, { P: tr("P production", "P 产量"), XG: tr("XG by-product", "XG 副产物"), R: tr("R recycle", "R 循环量") })}
  ${clockBlock(1056, 36, "end")}${condsAt(1056, 92, "end")}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, WO, "P", 0.01, 3, "kg/s"), o2 = tracked(q, s, tr, WO, "XG", 0.002, 4, ""), o3 = tracked(q, s, tr, WO, "R", 0.2, 2, "kg/s")
    const lim = limited(q, "F", s.v.F, null, 16, `${fx(s.v.F, 2)}`)
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || o3 || lim)
    setText(q, '[data-value="fa"]', `${fx(s.cond.FA, 3)} kg/s`)
    setMv(q, "u0", s.v.u0, 1.5, 6, `${fx(s.v.u0, 2)} kg/s`, 150); setMv(q, "u1", s.v.u1, 70, 100, `${fx(s.v.u1, 1)} °C`, 150); setMv(q, "u2", s.v.u2, 0.1, 0.6, fx(s.v.u2, 3), 170)
  },
  flows: (s) => ({ fb: s.v.u0 / 4, fa: (s.cond.FA ?? 1.8) / 2.5, out: s.v.F / 14, prod: s.v.P / 0.8, rec: s.v.R / 9, purge: s.v.u2 * 1.5 }),
}

/* ---------- MMA solution polymerisation CSTR (mmapoly.py) ---------- */
const MM: Ind = { NAMW: [880, 122, 170], T: [880, 202, 170] }
const mma: Screen = {
  cond: { F: [["monomer feed", "单体进料"], f3(" m³/h")], CIin: [["initiator concentration", "引发剂浓度"], f2(" kmol/m³")], fe: [["initiator efficiency", "引发效率"], f2("")], Tin: [["feed temperature", "进料温度"], f1(" K")], ua: [["jacket heat transfer", "夹套传热"], pct] },
  svg: (tr) => `${svgOpen(tr("MMA polymerisation process screen, replay", "MMA 聚合流程画面回放"), "mm-arrow")}
  <text class="hmi-note" x="520" y="28" text-anchor="middle">${tr("free-radical polymerisation · exothermic, steep in temperature", "自由基聚合 · 放热，对温度敏感")}</text>
  ${pipe("feed", "M150 110H470V128", "hmi-dr", "mm-arrow")}${readout("F", 150, 98, tr("F monomer", "F 单体进料"), 170)}<text class="hmi-sub" x="150" y="134">${tr("measured", "可测")}</text>
  ${pipe("ini", "M150 176H438", "sh-u1", "mm-arrow")}${valve(320, 176)}${mv("u0", 150, 202, tr("FI initiator", "FI 引发剂"), 170)}
  ${cstr("R", 440, 130, 160, 170, true, tr("CSTR", "反应釜"))}
  ${sensor("T", 560, 230, "TT")}
  ${pipe("jk", "M360 290H426", "sh-u2", "mm-arrow")}${mv("u1", 150, 262, tr("Tj jacket temperature", "Tj 夹套温度"), 200)}
  <text class="hmi-sub" x="150" y="306">${tr("base layer holds it", "基础层保持")}</text>
  ${readout("Pr", 440, 340, tr("Polymer", "聚合物"), 160)}<text class="hmi-sub" x="440" y="358">${tr("kg/h, monitored", "kg/h，只监视")}</text>
  ${pipe("prod", "M600 250H860", "hmi-dr", "mm-arrow")}${sensor("NAMW", 720, 226, "AT", "M720 250V241")}
  ${readout("X", 880, 282, tr("X ≤ 0.13", "X 转化率 ≤ 0.13"), 170)}<text class="hmi-sub" x="880" y="300">${tr("conversion, viscosity limit", "黏度限值")}</text>
  ${inds(MM, { NAMW: tr("NAMW", "NAMW"), T: tr("T reactor", "T 反应温度") })}
  ${clockBlock(1056, 36, "end")}${condsAt(24, 40)}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, MM, "NAMW", 500, 0, "kg/kmol"), o2 = tracked(q, s, tr, MM, "T", 0.5, 1, "K")
    const lim = limited(q, "X", s.v.X, null, 0.13, fx(s.v.X, 4))
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || lim)
    setText(q, '[data-value="F"]', `${fx(s.v.F, 3)} m³/h`); setText(q, '[data-value="Pr"]', fx(s.v.Pr, 1))
    setMv(q, "u0", s.v.u0, 0.004, 0.05, `${fx(s.v.u0, 4)} m³/h`, 170); setMv(q, "u1", s.v.u1, 300, 345, `${fx(s.v.u1, 1)} K`, 200)
  },
  flows: (s) => ({ feed: s.v.F, ini: s.v.u0 / 0.03, prod: s.v.F, jk: 0.5 }),
}

/* ---------- MSMPR cooling crystallizer with fines dissolution (crystallizer.py) ---------- */
const CR: Ind = { L43: [890, 120, 160], prod: [890, 210, 160] }
const cryst: Screen = {
  cond: { Q: [["feed flow", "进料流量"], f2(" m³/h")], Cin: [["feed concentration", "进料浓度"], (v: number) => `${fx(v, 0)} kg/m³`], kg: [["growth rate", "生长速率"], (v: number) => pct(v / 72)], kb: [["nucleation rate", "成核速率"], (v: number) => pct(v / 3150)] },
  svg: (tr) => `${svgOpen(tr("Crystallizer process screen, replay", "结晶器流程画面回放"), "cr-arrow")}
  <text class="hmi-note" x="520" y="28" text-anchor="middle">${tr("cooling crystallisation · moments of the size distribution", "冷却结晶 · 粒度分布的矩模型")}</text>
  ${pipe("feed", "M150 110H470V128", "hmi-dr", "cr-arrow")}${readout("Q", 150, 98, tr("Q feed", "Q 进料"), 170)}<text class="hmi-sub" data-value="feedc" x="150" y="134"></text>
  ${cstr("R", 440, 130, 160, 170, true, tr("Crystallizer", "结晶器"))}
  ${pipe("fines", "M600 170H700V100H540V128", "sh-u2", "cr-arrow")}${hxc(700, 136, 16)}<text class="hmi-sub" x="722" y="140">${tr("fines dissolver", "细晶溶解")}</text>
  ${mv("u1", 690, 200, tr("φ fines destroyed", "φ 细晶溶解比"), 150)}
  ${pipe("jk", "M360 290H426", "sh-u1", "cr-arrow")}${mv("u0", 150, 262, tr("T crystallizer", "T 结晶温度"), 200)}
  <text class="hmi-sub" x="150" y="306">${tr("base layer holds it", "基础层保持")}</text>
  ${readout("MT", 440, 340, tr("MT magma", "MT 晶浆"), 160)}<text class="hmi-sub" x="440" y="358">${tr("kg/m³, monitored", "kg/m³，只监视")}</text>
  ${pipe("prod", "M600 260H870", "hmi-dr", "cr-arrow")}${sensor("L43", 720, 236, "AT", "M720 260V251")}${sensor("prod", 800, 236, "FT", "M800 260V251")}
  ${readout("S", 890, 300, tr("S ≤ 0.06", "S 过饱和度 ≤ 0.06"), 160)}<text class="hmi-sub" x="890" y="318">${tr("supersaturation, metastable limit", "介稳区上限")}</text>
  ${inds(CR, { L43: tr("L43 mean size", "L43 平均粒度"), prod: tr("Production", "产量") })}
  ${clockBlock(1056, 36, "end")}${condsAt(24, 40)}
  </svg>`,
  update: (q, s, tr) => {
    const o1 = tracked(q, s, tr, CR, "L43", 3, 1, "µm"), o2 = tracked(q, s, tr, CR, "prod", 2, 1, "kg/h")
    const lim = limited(q, "S", s.v.S, null, 0.06, fx(s.v.S, 4))
    q('[data-vessel="R"]')?.classList.toggle("warn", o1 || o2 || lim)
    setText(q, '[data-value="Q"]', `${fx(s.v.Q, 2)} m³/h`); setText(q, '[data-value="MT"]', fx(s.v.MT, 1))
    setText(q, '[data-value="feedc"]', tr(`Cin ${fx(s.cond.Cin, 0)} kg/m³ · measured`, `Cin ${fx(s.cond.Cin, 0)} kg/m³ · 可测`))
    setMv(q, "u0", s.v.u0, 10, 45, `${fx(s.v.u0, 1)} °C`, 200); setMv(q, "u1", s.v.u1, 0, 0.9, fx(s.v.u1, 3), 150)
  },
  flows: (s) => ({ feed: s.v.Q / 2, fines: 0.2 + s.v.u1, prod: s.v.Q / 2, jk: 0.5 }),
}

/* ---------- literature multiloop columns (lit_batch.py, docs/heldout.md section 3): the scenario's model, its 2 or 3 loops ----------
   One column; each MV enters on the left at its loop's height and its paired CV leaves on the right at the same height (CVk with
   MVk, the literature pairing). The models' channels are anonymised in the packet (normalised units, rows shuffled), so the screen
   names the model, not the streams; measured disturbances read out below. Facts from docs/heldout.md. */
const LIT_INFO: Record<string, { name: string; kind: [string, string]; fact: [string, string] }> = {
  wb: { name: "Wood–Berry", kind: ["methanol–water", "甲醇–水塔"], fact: ["RGA 2.01 · θ 1–7 min", "RGA 2.01 · 滞后 1–7 分钟"] },
  ory: { name: "Ogunnaike–Ray", kind: ["ethanol–water", "乙醇–水塔"], fact: ["θ up to 9.4 min", "滞后最长 9.4 分钟"] },
  ty: { name: "Tyreus", kind: ["stabiliser", "稳定塔"], fact: ["one θ of 60 min", "一路滞后 60 分钟"] },
  al: { name: "Alatiqi–Luyben", kind: ["side-stream column", "侧线塔"], fact: ["θ up to 17 min", "滞后最长 17 分钟"] },
}
const LIT_COL = { x: 430, y: 56, w: 180, h: 272 }
function litGeo(d: Layout) {
  const n = d.mvs.length, ys = n >= 3 ? [124, 212, 300] : [134, 282]   // below the clock block
  const loops = d.mvs.map((m, k) => ({ m, k, y: ys[Math.min(k, ys.length - 1)], cv: d.cvs.find((c) => c.tracked && (c.name || c.key) === m.paired) }))
  const dist = d.cvs.filter((c) => !c.tracked)
  return { loops, dist }
}
const lit: Screen = {
  cond: {},
  conds: (c, tr) => [tr(`Literature column · ${c.dt} s step`, `文献精馏塔 · 每步 ${c.dt} 秒`),
    c.nchg ? tr(`plant changes so far: ${c.nchg}`, `已发生装置变化 ${c.nchg} 次`) : tr("plant as at the start", "装置与起点相同")],
  svg: (tr, d) => {
    const { x, y, w, h } = LIT_COL, info = LIT_INFO[d.lit?.model || ""], cx = x + w / 2, n = d.mvs.length
    const lab = (c: { label: string; label_zh?: string }) => esc(tr(c.label, c.label_zh || c.label))
    const { loops, dist } = litGeo(d)
    const mvs = loops.map(({ m, k, y: ly }) => `${pipe(`m${k}`, `M40 ${ly}H${x - 2}`, MVC[k % MVC.length], "li-arrow")}${valve(330, ly)}${mv(m.key, 40, ly - 22, lab(m), 220)}`).join("")
    const cvs = loops.map(({ k, y: ly, cv }) => cv ? `${pipe(`c${k}`, `M${x + w + 2} ${ly}H748`, "hmi-dr", "li-arrow")}${sensor(cv.name || cv.key, 690, ly - 22, "AT", `M690 ${ly}V${ly - 7}`)}${indicator(cv.name || cv.key, 760, ly - 14, 280, lab(cv))}` : "").join("")
    const ds = dist.map((c, i) => readout(c.name || c.key, 760 + i * 150, 352, tr(`D${i + 1}, measured`, `实测扰动 D${i + 1}`), 130)).join("")
    return `${svgOpen(tr("Literature column process screen, replay", "文献精馏塔流程画面回放"), "li-arrow")}
  ${pipe("vap", `M${cx} ${y}V34H${cx + 88}`, "hmi-dr", "li-arrow")}${hxc(cx + 108, 34, 16)}<text class="hmi-sub" x="${cx + 132}" y="38">${tr("condenser", "冷凝器")}</text>
  ${pipe("liq", `M${cx} ${y + h}V352H${cx + 88}`, "hmi-dr", "li-arrow")}${hxc(cx + 108, 352, 16)}<text class="hmi-sub" x="${cx + 132}" y="356">${tr("reboiler", "再沸器")}</text>
  <rect class="hmi-vessel-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="30"/>
  <path class="tep-tray" d="${[0.12, 0.24, 0.76, 0.88].map((f) => `M${x + 12} ${(y + f * h).toFixed(0)}H${x + w - 12}`).join("")}" stroke-dasharray="4 4"/>
  <rect class="hmi-vessel" data-vessel="C" x="${x}" y="${y}" width="${w}" height="${h}" rx="30"/>
  ${info ? `<text class="hmi-tag" x="${cx}" y="${y + h / 2 - 30}" text-anchor="middle">${info.name}</text>
  <text class="hmi-sub" x="${cx}" y="${y + h / 2 - 8}" text-anchor="middle">${tr(`${info.kind[0]} · ${n}×${n}`, `${info.kind[1]} · ${n}×${n}`)}</text>
  <text class="hmi-sub" x="${cx}" y="${y + h / 2 + 14}" text-anchor="middle">${tr(info.fact[0], info.fact[1])}</text>
  <text class="hmi-sub" x="${cx}" y="${y + h / 2 + 36}" text-anchor="middle">${tr("±20% variant · hidden", "±20% 变体 · 模型不公开")}</text>` : ""}
  ${mvs}${cvs}${ds}
  ${clockBlock(24, 36)}${condsAt(24, 350)}
  </svg>`
  },
  update: (q, s, tr, d) => {
    let bad = false
    const { loops, dist } = litGeo(d)
    for (const { y: ly, cv } of loops) {
      if (!cv) continue
      const k = cv.name || cv.key, v = s.v[k], r = s.ref[k], b = cv.band || 0
      setIndicator(q, k, 760, 280, ly - 14, v, r, b, fx(v, cv.dec), tr(`target ${fx(r, cv.dec)} ± ${fx(b, cv.dec)}`, `目标 ${fx(r, cv.dec)} ± ${fx(b, cv.dec)}`))
      const out = Math.abs(v - r) > b
      q(`[data-at="${k}"]`)?.classList.toggle("warn", out); bad = bad || out
    }
    for (const c of dist) setText(q, `[data-value="${c.name || c.key}"]`, fx(s.v[c.name || c.key], c.dec))
    q('[data-vessel="C"]')?.classList.toggle("warn", bad)
    for (const m of d.mvs) setMv(q, m.key, s.v[m.key], m.lo ?? 0, m.hi ?? 1, fx(s.v[m.key], m.dec), 220)
  },
  flows: (s, d) => {
    const out: Record<string, number> = { vap: 0.4, liq: 0.4 }
    d.mvs.forEach((m, k) => { out[`m${k}`] = 0.2 + 0.8 * ((s.v[m.key] - (m.lo ?? 0)) / ((m.hi ?? 1) - (m.lo ?? 0))); out[`c${k}`] = 0.4 })
    return out
  },
}

/* ---------- the unified simulation core (aio/sim, docs/sim.md): a new plant every episode, drawn from the file ----------
   As the random-plant screen: the MVs on the left in the MV colours, the plant as one block that names its structure and
   hides its model, the tracked CVs (indicator, band around the reference, priority) and zone CVs (readout, current limits)
   on the right, the measured disturbances entering from the top. Values are in % of range; zone limits, priorities and
   the operator's MV limits move with the events (snapshot conditions lo_/hi_/w_<cv>, ulo_/uhi_<mv>: ticks on the MV
   bar). At a constrained operating point a tracked CV may sit anywhere from its target to the reachable optimum, a
   band beyond (oklo_/okhi_<cv>): the indicator's shaded band widens to that stretch. The screen grows with the rows. */
const SIM_NONE = 1e8   // research_site_sim_replay.NONE: no limit on that side
const simLim = (v: number | undefined) => (v == null || Math.abs(v) >= SIM_NONE ? null : v)
function simGeo(d: Layout) {
  const out = d.cvs.filter((c) => c.tracked || c.zone), dv = d.cvs.filter((c) => !c.tracked && !c.zone)
  const rows: { c: (typeof out)[number]; y: number }[] = []
  let y = 44
  for (const c of out) { rows.push({ c, y }); y += c.tracked ? 58 : 44 }
  const H = Math.max(384, y + 56), top = 82, bottom = H - 70
  const mys = d.mvs.map((_, j) => top + 18 + ((j + 0.5) * (bottom - top - 36)) / d.mvs.length - 22)
  return { rows, dv, H, top, bottom, mys }
}
const simLimText = (lo: number | null, hi: number | null, dec: number, tr: Tr) =>
  lo != null && hi != null ? tr(`limits ${fx(lo, dec)}–${fx(hi, dec)}%`, `限值 ${fx(lo, dec)}–${fx(hi, dec)}%`)
  : hi != null ? tr(`at most ${fx(hi, dec)}%`, `不高于 ${fx(hi, dec)}%`) : lo != null ? tr(`at least ${fx(lo, dec)}%`, `不低于 ${fx(lo, dec)}%`) : tr("no limit", "无限值")
const sim: Screen = {
  cond: {},
  conds: (c, tr) => [tr(`Simulation core · one decision every ${c.dt} s`, `新仿真核心 · 每 ${c.dt} 秒决策一次`),
    c.nchg ? tr(`plant changes so far: ${c.nchg}`, `已发生装置变化 ${c.nchg} 次`) : tr("plant as at the start", "装置与起点相同")],
  svg: (tr, d) => {
    const { rows, dv, H, top, bottom, mys } = simGeo(d), st = d.start || {}, lab = (c: { label: string; label_zh?: string }) => esc(tr(c.label, c.label_zh || c.label))
    const tick = (k: string, side: string, y: number) => `<line class="hmi-limit" data-${side}="${k}" x1="40" x2="40" y1="${y + 4}" y2="${y + 16}" visibility="hidden"/>`
    const mvs = d.mvs.map((m, j) => `${pipe(`m${j}`, `M40 ${mys[j] + 22}H398`, MVC[j % MVC.length], "sm-arrow")}${valve(300, mys[j] + 22)}${mv(m.key, 40, mys[j], lab(m), 200)}`
      + tick(m.key, "ulo", mys[j]) + tick(m.key, "uhi", mys[j])).join("")
    const cvs = rows.map(({ c, y }) => pipe(`c-${c.name}`, `M622 ${y + 4}H740`, "hmi-dr", "sm-arrow")
      + (c.tracked ? indicator(c.name!, 760, y, 280, lab(c)) : readout(c.name!, 760, y, lab(c), 280, simLimText(simLim(st[`lo_${c.name}`]), simLim(st[`hi_${c.name}`]), c.dec, tr)))).join("")
    const dvs = dv.map((c, i) => { const x = 440 + i * 110; return `${pipe(`d-${c.name}`, `M${x + 30} 50V${top - 2}`, "hmi-dr", "sm-arrow")}${readout(c.name!, x, 40, esc(c.tag || ""), 90)}` }).join("")
    const cx = 510, mid = (top + bottom) / 2, sp = d.sim
    const nl = st.linear ? tr("linear", "线性") : tr("nonlinear", "非线性")
    const lines = [
      tr(`${st.nm} MVs → ${st.nt} tracked${st.nz ? `, ${st.nz} zone` : ""} CVs`, `${st.nm} 个 MV → ${st.nt} 个跟踪${st.nz ? `、${st.nz} 个区间` : ""} CV`),
      tr(`coupling: ${sp?.arch_label || "—"}`, `耦合结构：${sp?.arch_label_zh || sp?.arch_label || "—"}`),
      tr(`time scale ${fx(st.T, 0)} steps`, `时间尺度 ${fx(st.T, 0)} 步`),
      tr(`dead time / lag ${fx(st.dead, 2)}`, `纯滞后 / 时间常数 ${fx(st.dead, 2)}`),
      st.load ? tr(`${nl}, load-dependent`, `${nl}、随负荷变化`) : nl,
      sp?.nq ? tr(`${sp.nq} unmeasured load${sp.nq > 1 ? "s" : ""} · model hidden`, `${sp.nq} 个不可测负荷 · 模型不公开`) : tr("model hidden", "模型不公开"),
    ]
    return `<svg class="hmi-svg" viewBox="0 0 1080 ${H}" role="img" aria-label="${esc(tr("Simulation-core plant process screen, replay", "新仿真核心装置流程画面回放"))}"><defs>${arrowDef("sm-arrow")}</defs>
  <rect class="hmi-vessel-bg" x="400" y="${top}" width="220" height="${bottom - top}" rx="12"/><rect class="hmi-vessel" data-vessel="P" x="400" y="${top}" width="220" height="${bottom - top}" rx="12"/>
  <text class="hmi-tag" x="${cx}" y="${mid - 58}" text-anchor="middle">${d.plant_label ? esc(tr(d.plant_label, d.plant_label_zh || d.plant_label)) : tr("New plant", "新装置")}</text>
  ${lines.map((t, i) => `<text class="hmi-sub" x="${cx}" y="${mid - 34 + i * 18}" text-anchor="middle">${esc(t)}</text>`).join("")}
  ${dv.length ? `<text class="hmi-sub" x="${440 + dv.length * 110}" y="40">${tr("measured", "实测扰动")}</text>` : ""}
  ${mvs}${cvs}${dvs}
  ${clockBlock(24, 40)}${condsAt(24, H - 34)}
  </svg>`
  },
  update: (q, s, tr, d) => {
    let bad = false
    for (const { c, y } of simGeo(d).rows) {
      const k = c.name!, v = s.v[k]
      if (c.tracked) {
        const r = s.ref[k], b = c.band || 0, w = s.cond[`w_${k}`], ol = s.cond[`oklo_${k}`], oh = s.cond[`okhi_${k}`]
        /* a constrained point: the stretch that does not count as outside, where it is wider than the band */
        const wide = ol != null && oh != null && (r - b - ol > 0.5 * 10 ** -c.dec || oh - r - b > 0.5 * 10 ** -c.dec)
        const lo = wide ? ol : r - b, hi = wide ? oh : r + b, out = v < lo || v > hi
        const pri = w != null ? tr(` · priority ${fx(w, 2)}`, ` · 优先级 ${fx(w, 2)}`) : ""
        setIndicator(q, k, 760, 280, y, v, r, b, `${fx(v, c.dec)}%`, wide
          ? tr(`target ${fx(r, c.dec)} · allowed ${fx(lo, Math.max(c.dec - 1, 0))}–${fx(hi, Math.max(c.dec - 1, 0))}`, `目标 ${fx(r, c.dec)} · 容许 ${fx(lo, Math.max(c.dec - 1, 0))}–${fx(hi, Math.max(c.dec - 1, 0))}`) + pri
          : tr(`target ${fx(r, c.dec)} ± ${fx(b, c.dec)}`, `目标 ${fx(r, c.dec)} ± ${fx(b, c.dec)}`) + pri)
        q(`[data-value="${k}"]`)?.classList.toggle("warn", out); q(`[data-pointer="${k}"]`)?.classList.toggle("warn", out)
        /* the rail spans ±5 bands around the reference; the shaded band is the stretch inside it */
        const px = (x: number) => 760 + 140 + Math.min(Math.max((x - r) / (5 * b), -1), 1) * 140, band = q(`[data-band="${k}"]`)
        if (band && b > 0) { band.setAttribute("x", px(lo).toFixed(1)); band.setAttribute("width", Math.max(2, px(hi) - px(lo)).toFixed(1)) }
        bad = bad || out
      } else {
        const lo = simLim(s.cond[`lo_${k}`]), hi = simLim(s.cond[`hi_${k}`])
        bad = limited(q, k, v, lo, hi, `${fx(v, c.dec)}%`) || bad
        setText(q, `[data-sub="${k}"]`, simLimText(lo, hi, c.dec, tr))
      }
    }
    for (const c of simGeo(d).dv) setText(q, `[data-value="${c.name}"]`, `${fx(s.v[c.name!], c.dec)}%`)
    q('[data-vessel="P"]')?.classList.toggle("warn", bad)
    for (const m of d.mvs) {
      setMv(q, m.key, s.v[m.key], m.lo ?? 0, m.hi ?? 100, `${fx(s.v[m.key], m.dec)}%`, 200)
      for (const side of ["ulo", "uhi"]) {
        const v = s.cond[`${side}_${m.key}`], e = q(`[data-${side}="${m.key}"]`); if (!e) continue
        if (v == null) { e.setAttribute("visibility", "hidden"); continue }
        const x = (40 + 2 * Math.min(Math.max(v, 0), 100)).toFixed(1)
        e.setAttribute("x1", x); e.setAttribute("x2", x); e.setAttribute("visibility", "visible")
      }
    }
  },
  flows: (s, d) => {
    const out: Record<string, number> = {}
    d.mvs.forEach((m, j) => { out[`m${j}`] = 0.2 + 0.8 * ((s.v[m.key] - (m.lo ?? 0)) / ((m.hi ?? 100) - (m.lo ?? 0))) })
    for (const c of d.cvs) { out[`c-${c.name}`] = 0.4; out[`d-${c.name}`] = 0.4 }
    return out
  },
}

export const ZOO: Record<string, Screen> = { proc, proc2, mimo, net, evap, vdv, excstr, bio, column, hex, furnace, gasnet, boiler, wo, mma, cryst, lit, sim }
