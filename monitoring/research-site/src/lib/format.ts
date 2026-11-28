import type { Lang } from "./i18n"

/* Numbers are rounded, never truncated (AGENTS.md rule 10). */
export const number = (n: number) => new Intl.NumberFormat("en-US").format(n)
export const round = (n: number | null | undefined) => (n == null || !Number.isFinite(n) ? "—" : number(Math.round(n)))
export const fx = (v: number | null | undefined, d: number) => (v == null || !Number.isFinite(v) ? "—" : v.toFixed(d))
/* a multiple of the formula PI to two significant figures: 0.096×, 0.71×, 3.8×, 12× (half up, rule 10) */
export const times = (y: number) => { const k = 10 ** Math.max(0, 1 - Math.floor(Math.log10(y))); return `${number(Math.round(y * k + 1e-9) / k)}×` }
export const mega = (v: number | null | undefined) =>
  v == null ? "—" : v >= 1e9 ? `${(v / 1e9).toFixed(2)}B` : v >= 1e6 ? `${(v / 1e6).toFixed(1)}M` : number(Math.round(v))
export const short = (n: number) =>
  Math.abs(n) >= 1e6 ? `${Number((n / 1e6).toFixed(1))}M` : Math.abs(n) >= 1e4 ? `${Math.round(n / 1000)}k` : Math.abs(n) >= 1000 ? `${Number((n / 1000).toFixed(1))}k` : Number(n.toPrecision(3)).toString()
export const mmss = (s: number) => {
  s = Math.max(0, Math.round(s))
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
}
export const hhmm = (s: number) => {
  s = Math.max(0, Math.round(s))
  return `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}`
}
const MON = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
/* '2026-11-03T22:31+08:00' or '2026-11-03 22:31' -> '3 Nov 22:31' / '11月3日 22:31' (times are UTC+8 as written). */
export function when(iso: string | null | undefined, lang: Lang) {
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}:\d{2}))?/.exec(iso || "")
  if (!m) return iso || "—"
  return lang === "zh" ? `${+m[2]}月${+m[3]}日${m[4] ? ` ${m[4]}` : ""}` : `${+m[3]} ${MON[+m[2] - 1]}${m[4] ? ` ${m[4]}` : ""}`
}
export const esc = (s: unknown) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c] as string)

/* Key point only (user 11-10): the first clause of a note, cut at the first "；" "。" "; " or ". " outside brackets
   (not inside "0.63" or "(2,756; 30 of 32)"). */
export function brief(s: string | undefined | null): string {
  if (!s) return ""
  let depth = 0
  for (let i = 0; i < s.length; i++) {
    const c = s[i], next = s[i + 1]
    if (c === "(" || c === "（") depth++
    else if ((c === ")" || c === "）") && depth > 0) depth--
    else if (depth === 0 && (c === "；" || c === "。" || ((c === ";" || c === ".") && (next === " " || next === undefined))))
      return s.slice(0, i).replace(/[，,：:]\s*$/, "").trim()
  }
  return s.trim()
}
