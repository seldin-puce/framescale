import { ChevronDown, Cpu, Leaf, Sparkles } from "lucide-react";
import { cn } from "@/lib/cn";
import { useApp } from "@/stores/app";
import { useTelemetry } from "@/stores/telemetry";
import { backend } from "@/ipc";
import { Tooltip } from "../ui/Tooltip";

const isMac = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);

export function TitleBar() {
  const hw = useApp((s) => s.hardware);
  const setDevice = useApp((s) => s.setDevice);
  const tier = useApp((s) => s.entitlements.tier);
  const devKey = useApp((s) => s.developmentKey);
  const openPro = useApp((s) => s.openPro);
  const eco = useApp((s) => s.eco);
  const t = useTelemetry();

  return (
    <header
      data-tauri-drag-region
      className={cn("flex h-11 shrink-0 items-center gap-3 border-b border-border bg-surface-glass pr-3", isMac ? "pl-20" : "pl-3")}
    >
      <div data-tauri-drag-region className="flex items-center gap-2">
        <span className="grid size-5 place-items-center rounded-[5px] bg-gradient-to-br from-accent to-[#8fb8ff] text-[10px] font-black text-white">F</span>
        <span data-tauri-drag-region className="text-[13px] font-semibold tracking-tight">FrameScale AI</span>
        {backend.kind === "demo" && (
          <Tooltip label="Running in a browser: hardware, jobs and telemetry are simulated.">
            <span className="rounded-[4px] bg-white/8 px-1.5 py-0.5 text-[10px] font-medium text-fg-muted ring-1 ring-border">Demo data</span>
          </Tooltip>
        )}
      </div>

      <div data-tauri-drag-region className="flex-1" />

      <label className="relative flex items-center gap-1.5 rounded-sm bg-black/25 py-1 pr-6 pl-2 ring-1 ring-border hover:ring-border-strong">
        <Cpu className="size-3.5 text-fg-subtle" />
        <span className="sr-only">Processing device</span>
        <select
          className="max-w-56 appearance-none truncate bg-transparent text-xs text-fg outline-none"
          value={hw?.selected ?? ""}
          onChange={(e) => void setDevice(e.target.value || null)}
          disabled={!hw?.devices.length}
        >
          {!hw?.devices.length && <option value="">{hw?.ready ? "CPU only" : "Detecting GPUs…"}</option>}
          {hw?.devices.map((d) => (
            <option key={d.uuid} value={d.uuid}>
              {d.name}{d.kind === "integrated" ? " (iGPU)" : ""}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-1.5 size-3 text-fg-subtle" />
      </label>

      {eco.mode !== "off" && (
        <Tooltip label={`Eco mode · GPU duty ${t ? Math.round(t.eco.duty * 100) : 100}%`}>
          <span className="flex items-center gap-1 rounded-sm bg-ok/10 px-1.5 py-1 text-[11px] font-medium text-ok ring-1 ring-ok/25">
            <Leaf className="size-3" /> Eco
          </span>
        </Tooltip>
      )}

      {tier === "pro" ? (
        <Tooltip label={devKey ? "Pro (development key build)" : "FrameScale Pro"}>
          <span className="rounded-[4px] bg-pro px-1.5 py-0.5 text-[10px] font-bold tracking-wider text-black">PRO</span>
        </Tooltip>
      ) : (
        <button
          type="button"
          onClick={() => openPro(null)}
          className="flex items-center gap-1 rounded-sm px-2 py-1 text-xs font-medium text-pro ring-1 ring-pro/35 transition-colors hover:bg-pro/10"
        >
          <Sparkles className="size-3.5" /> Upgrade
        </button>
      )}
    </header>
  );
}
