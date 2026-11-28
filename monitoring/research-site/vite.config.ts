import path from "node:path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

/* The site is served from https://m3rcyzzz.club/aio/ with its data files (progress.json, replay.json,
   tep-replay.json, ph-replay.json, live/*) next to index.html, so every URL is relative. */
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": path.resolve(import.meta.dirname, "./src") } },
  build: { outDir: "dist", emptyOutDir: true, assetsInlineLimit: 0, chunkSizeWarningLimit: 600 },
})
