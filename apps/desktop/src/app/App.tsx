import { useEffect } from "react";
import { backend } from "@/ipc";
import { useApp } from "@/stores/app";
import { pushTelemetry } from "@/stores/telemetry";
import { AppShell } from "@/components/shell/AppShell";
import { TooltipProvider } from "@/components/ui/Tooltip";

// Dev builds only: expose the store for WebDriver end-to-end automation (native
// file dialogs can't be scripted). Stripped from production bundles.
if (import.meta.env.DEV) {
  (window as unknown as { __FS_STORE__: typeof useApp }).__FS_STORE__ = useApp;
}

// Module-level guard: StrictMode runs effects twice in development.
let seeded = false;

export function App() {
  useEffect(() => {
    const offs: (() => void)[] = [];
    let cancelled = false;
    void (async () => {
      await useApp.getState().init();
      const jobs = await backend.subscribeJobs((e) => useApp.getState().handleEvent(e));
      const tele = await backend.subscribeTelemetry(pushTelemetry);
      if (cancelled) {
        jobs();
        tele();
      } else offs.push(jobs, tele);
      // Demo-only: `?seed` pre-populates the queue for design review and screenshots.
      if (backend.kind === "demo" && !seeded && new URLSearchParams(location.search).has("seed")) {
        seeded = true;
        const [first, ...rest] = await backend.pickFiles();
        if (first) {
          await useApp.getState().addFiles([first]);
          await useApp.getState().renderAll();
        }
        await useApp.getState().addFiles(rest);
      }
      // The window is created hidden; reveal after the first paint (no white flash).
      requestAnimationFrame(() => void backend.showWindow());
    })();
    return () => {
      cancelled = true;
      offs.forEach((f) => f());
    };
  }, []);

  return (
    <TooltipProvider>
      <AppShell />
    </TooltipProvider>
  );
}
