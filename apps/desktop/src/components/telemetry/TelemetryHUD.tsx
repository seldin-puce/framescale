import type { EcoMode, GpuDevice, Vendor } from "@framescale/ipc-types";
import { BatteryCharging, Cpu, HardDrive, Leaf, Thermometer, Zap } from "lucide-react";
import { useEffect, useRef } from "react";
import { cn } from "@/lib/cn";
import { DASH, bytes, fps as fmtFps, pct } from "@/lib/format";
import { useApp } from "@/stores/app";
import { CAPACITY_SAMPLES, series, useTelemetry } from "@/stores/telemetry";
import { Section } from "../ui/Section";
import { Segmented } from "../ui/Segmented";
import { Tooltip } from "../ui/Tooltip";

const VENDOR: Record<Vendor, { label: string; className: string }> = {
  nvidia: { label: "NVIDIA", className: "text-[#76b900] ring-[#76b900]/30 bg-[#76b900]/10" },
  amd: { label: "AMD", className: "text-[#ff5a5a] ring-[#ff5a5a]/30 bg-[#ff5a5a]/10" },
  intel: { label: "Intel", className: "text-[#4fa3ff] ring-[#4fa3ff]/30 bg-[#4fa3ff]/10" },
  apple: { label: "Apple", className: "text-fg ring-white/20 bg-white/8" },
  other: { label: "GPU", className: "text-fg-muted ring-white/15 bg-white/5" },
};

function Stat({ icon: Icon, label, value, tone }: { icon: typeof Cpu; label: string; value: string; tone?: "warn" | "err" | undefined }) {
  return (
    <div className="flex min-w-0 items-center gap-1.5 rounded-sm bg-black/20 px-2 py-1.5 ring-1 ring-border">
      <Icon className={cn("size-3.5 shrink-0 text-fg-subtle", tone === "warn" && "text-warn", tone === "err" && "text-err")} strokeWidth={1.75} />
      <div className="min-w-0">
        <div className="text-[10px] leading-none text-fg-subtle">{label}</div>
        <div className={cn("tabular mt-0.5 font-mono text-xs leading-none text-fg", tone === "warn" && "text-warn", tone === "err" && "text-err")}>{value}</div>
      </div>
    </div>
  );
}

function DeviceCard({ device, error }: { device: GpuDevice | undefined; error: string | null }) {
  if (!device) {
    return (
      <div className="rounded-sm bg-black/20 p-2 text-xs text-fg-muted ring-1 ring-border">
        {error ? `No Vulkan GPU found (${error}). Processing will use the CPU.` : "Detecting GPUs…"}
      </div>
    );
  }
  const v = VENDOR[device.vendor];
  return (
    <div className="flex items-center gap-2">
      <span className={cn("rounded-[4px] px-1.5 py-0.5 text-[10px] font-bold tracking-wide ring-1", v.className)}>{v.label}</span>
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-medium" title={device.name}>{device.name}</div>
        <div className="truncate font-mono text-[10px] text-fg-subtle">
          Vulkan {device.apiVersion} · {device.fp16 ? "FP16" : "FP32"}
          {device.cooperativeMatrix ? " · Tensor/XMX" : ""} · {device.kind}
        </div>
      </div>
    </div>
  );
}

function MemoryBar({ used, total, unified }: { used: number | null; total: number | null; unified: boolean }) {
  const ratio = used != null && total ? Math.min(1, used / total) : null;
  const tone = ratio == null ? "bg-white/10" : ratio > 0.92 ? "bg-err" : ratio > 0.8 ? "bg-warn" : "bg-accent";
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[11px]">
        <span className="text-fg-muted">{unified ? "Unified memory" : "VRAM"}</span>
        <span className="tabular font-mono text-fg">
          {bytes(used)} <span className="text-fg-subtle">/ {bytes(total, 0)}</span>
        </span>
      </div>
      <div
        className="h-2 overflow-hidden rounded-full bg-black/40 ring-1 ring-border"
        role="meter"
        aria-label={unified ? "Unified memory in use" : "VRAM in use"}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={ratio == null ? undefined : Math.round(ratio * 100)}
      >
        <div className={cn("h-full rounded-full transition-[width] duration-500 ease-out", tone)} style={{ width: `${(ratio ?? 0) * 100}%` }} />
      </div>
    </div>
  );
}

/** Canvas sparkline fed straight from the telemetry ring (no React state per sample). */
function FpsSparkline({ version }: { version: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = (c.width = Math.round(c.clientWidth * dpr));
    const h = (c.height = Math.round(c.clientHeight * dpr));
    ctx.clearRect(0, 0, w, h);
    const data = series((t) => t.renderFps);
    const max = Math.max(1, ...data.map((v) => v ?? 0)) * 1.15;
    const x = (i: number) => w - (data.length - 1 - i) * (w / (CAPACITY_SAMPLES - 1));
    const y = (v: number) => h - (v / max) * (h - 2 * dpr) - dpr;

    const path = new Path2D();
    let open = false;
    data.forEach((v, i) => {
      if (v == null) {
        open = false;
        return;
      }
      if (open) path.lineTo(x(i), y(v));
      else path.moveTo(x(i), y(v));
      open = true;
    });
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(76,141,255,0.35)");
    grad.addColorStop(1, "rgba(76,141,255,0)");
    ctx.lineWidth = 1.5 * dpr;
    ctx.strokeStyle = "#4c8dff";
    ctx.lineJoin = "round";
    ctx.stroke(path);
    // Area fill under the most recent contiguous run.
    const lastNull = data.lastIndexOf(null);
    const run = data.slice(lastNull + 1) as number[];
    if (run.length > 1) {
      const area = new Path2D();
      const off = lastNull + 1;
      area.moveTo(x(off), h);
      run.forEach((v, i) => area.lineTo(x(off + i), y(v)));
      area.lineTo(x(data.length - 1), h);
      area.closePath();
      ctx.fillStyle = grad;
      ctx.fill(area);
    }
  }, [version]);
  return <canvas ref={ref} className="h-10 w-full" aria-hidden />;
}

const ECO_OPTIONS = [
  { value: "off", label: "Off" },
  { value: "auto", label: "Auto" },
  { value: "50", label: "50%" },
] as const;

function ecoValue(m: EcoMode): (typeof ECO_OPTIONS)[number]["value"] {
  return m.mode === "off" ? "off" : m.mode === "auto" ? "auto" : "50";
}

export function TelemetryHUD() {
  const t = useTelemetry();
  const hw = useApp((s) => s.hardware);
  const eco = useApp((s) => s.eco);
  const setEco = useApp((s) => s.setEco);
  const device = hw?.devices.find((d) => d.uuid === hw.selected) ?? hw?.devices[0];
  const d = t?.device;
  const temp = d?.temperatureC ?? null;
  const tempTone = temp == null ? undefined : temp >= 85 ? "err" : temp >= 78 ? "warn" : undefined;
  const disk = t?.diskFreeBytes ?? null;

  return (
    <Section
      title="Hardware"
      aside={
        t?.eco.onBattery ? (
          <Tooltip label="On battery: Eco Auto caps GPU duty">
            <BatteryCharging className="size-3.5 text-warn" />
          </Tooltip>
        ) : null
      }
    >
      <div className="flex flex-col gap-3">
        <DeviceCard device={device} error={hw?.discoveryError ?? null} />
        <MemoryBar used={d?.vramUsed ?? null} total={d?.vramTotal ?? device?.vramBytes ?? null} unified={device?.unifiedMemory ?? false} />

        <div>
          <div className="mb-1 flex items-baseline justify-between text-[11px]">
            <span className="text-fg-muted">Render speed</span>
            <span className="tabular font-mono text-fg">
              {t?.renderFps != null ? (
                <>
                  <span className="text-[15px] font-semibold">{fmtFps(t.renderFps)}</span> <span className="text-fg-subtle">fps</span>
                </>
              ) : (
                <span className="text-fg-subtle">idle</span>
              )}
            </span>
          </div>
          <div className="rounded-sm bg-black/20 ring-1 ring-border">
            <FpsSparkline version={t?.atMs ?? 0} />
          </div>
        </div>

        <div className="grid grid-cols-3 gap-1.5">
          <Stat icon={Cpu} label="GPU load" value={pct(d?.utilizationPct)} />
          <Stat icon={Thermometer} label="Temp" value={temp == null ? DASH : `${Math.round(temp)}°C`} tone={tempTone} />
          <Stat icon={Zap} label="Power" value={d?.powerW == null ? DASH : `${Math.round(d.powerW)} W`} />
        </div>

        <div className="flex items-center gap-2">
          <Tooltip label="Eco caps average GPU duty with micro-sleeps between frames. It prevents laptop overheating on long batches. Auto targets 80 °C.">
            <span className="flex items-center gap-1 text-[11px] text-fg-muted">
              <Leaf className={cn("size-3.5", eco.mode !== "off" ? "text-ok" : "text-fg-subtle")} /> Eco
            </span>
          </Tooltip>
          <Segmented
            ariaLabel="Eco mode"
            className="flex-1"
            value={ecoValue(eco)}
            options={[...ECO_OPTIONS]}
            onChange={(v) => void setEco(v === "off" ? { mode: "off" } : v === "auto" ? { mode: "auto", targetC: 80 } : { mode: "fixed", dutyPct: 50 })}
          />
          <span className="tabular w-10 text-right font-mono text-[11px] text-fg-muted" title="Current GPU duty cycle">
            {t ? `${Math.round(t.eco.duty * 100)}%` : DASH}
          </span>
        </div>

        <div className="flex items-center justify-between rounded-sm bg-black/20 px-2 py-1.5 ring-1 ring-border">
          <span className="flex items-center gap-1.5 text-[11px] text-fg-muted">
            <HardDrive className={cn("size-3.5", disk != null && disk < 20 * 1024 ** 3 ? "text-warn" : "text-fg-subtle")} /> Free disk
          </span>
          <span className={cn("tabular font-mono text-xs", disk != null && disk < 20 * 1024 ** 3 ? "text-warn" : "text-fg")}>{bytes(disk, 0)}</span>
        </div>
      </div>
    </Section>
  );
}
