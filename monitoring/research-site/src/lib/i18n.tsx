import * as React from "react"

/* English / 简体中文. Every visible string is written as tr(en, zh) next to the code that shows it, so a
   missing translation is a type error; data files carry Chinese in sibling "<field>_zh" fields (pick).
   The language is chosen before first paint in index.html: ?lang=, then the saved choice, then the browser. */
export type Lang = "en" | "zh"

export function initialLang(): Lang {
  return document.documentElement.dataset.lang === "zh" ? "zh" : "en"
}

export type Tr = (en: string, zh: string) => string
export type Pick = (o: Record<string, unknown> | null | undefined, key: string) => string

export function makeTr(lang: Lang): Tr {
  return (en, zh) => (lang === "zh" ? zh : en)
}
export function makePick(lang: Lang): Pick {
  return (o, k) => {
    if (o == null) return ""
    const zh = o[`${k}_zh`]
    const v = lang === "zh" && zh != null ? zh : o[k]
    return v == null ? "" : String(v)
  }
}

type Ctx = { lang: Lang; tr: Tr; pick: Pick; setLang: (l: Lang) => void }
const LangContext = React.createContext<Ctx | null>(null)

export function LangProvider({ children }: { children: React.ReactNode }) {
  const [lang, setLangState] = React.useState<Lang>(initialLang)
  const setLang = React.useCallback((l: Lang) => {
    setLangState(l)
    try { localStorage.setItem("aio-lang", l) } catch { /* private mode: the URL keeps the choice */ }
    try { const u = new URL(location.href); u.searchParams.set("lang", l); history.replaceState(history.state, "", u) } catch { /* sandboxed */ }
  }, [])
  React.useEffect(() => {
    const d = document.documentElement
    d.lang = lang === "zh" ? "zh-CN" : "en"
    d.dataset.lang = lang
    document.title = lang === "zh" ? "AIOGym · 研究进展" : "AIOGym · Research"
  }, [lang])
  const value = React.useMemo(() => ({ lang, tr: makeTr(lang), pick: makePick(lang), setLang }), [lang, setLang])
  return <LangContext.Provider value={value}>{children}</LangContext.Provider>
}

export function useLang(): Ctx {
  const c = React.useContext(LangContext)
  if (!c) throw new Error("useLang outside LangProvider")
  return c
}
