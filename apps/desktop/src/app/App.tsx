import { useEffect } from "react";
import { isTauri } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";

/**
 * Phase 2 placeholder shell. It proves the toolchain end to end: Vite +
 * Tailwind v4 tokens + Tauri window reveal. Phase 3 replaces it with the
 * full AppShell (Queue / Viewport / Inspector).
 */
export function App() {
  useEffect(() => {
    // The window is created hidden; reveal it after the first paint so there's no white flash.
    if (!isTauri()) return;
    requestAnimationFrame(() => {
      void getCurrentWindow().show();
    });
  }, []);

  return (
    <main className="grid h-full place-items-center">
      <div className="rounded-lg border border-border bg-surface-glass px-8 py-6 text-center backdrop-blur-xl">
        <h1 className="text-lg font-semibold tracking-tight">FrameScale AI</h1>
        <p className="mt-1 text-sm text-fg-muted">Workspace scaffold — Phase 2</p>
        <p className="tabular mt-3 font-mono text-xs text-fg-subtle">
          VRAM 0.0 / 0.0 GB · 0.0 fps
        </p>
      </div>
    </main>
  );
}
