import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"
import { useSyncExternalStore } from "react"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/* true while the media query matches (phones: "(max-width: 639px)", Tailwind's sm breakpoint) */
export function useMedia(q: string): boolean {
  return useSyncExternalStore(
    (cb) => { if (typeof matchMedia !== "function") return () => {}; const m = matchMedia(q); m.addEventListener("change", cb); return () => m.removeEventListener("change", cb) },
    () => typeof matchMedia === "function" && matchMedia(q).matches,
    () => false)
}
export const PHONE = "(max-width: 639px)"
