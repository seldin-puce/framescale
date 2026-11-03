import type { StageKind } from "@framescale/ipc-types";
import { Blend, ImageUpscale, Layers, ScanFace, Waves } from "lucide-react";
import { Fragment } from "react";
import { cn } from "@/lib/cn";
import { COMING_SOON, STAGE_ORDER, useApp } from "@/stores/app";
import { Tooltip } from "../ui/Tooltip";

export const STAGE_META: Record<StageKind, { label: string; icon: typeof Layers; hint: string }> = {
  deinterlace: { label: "Deinterlace", icon: Layers, hint: "BWDIF" },
  denoise: { label: "Denoise", icon: Waves, hint: "NAFNet · Real-CUGAN" },
  upscale: { label: "Upscale", icon: ImageUpscale, hint: "Real-ESRGAN" },
  faceFix: { label: "Face Fix", icon: ScanFace, hint: "CodeFormer" },
  interpolate: { label: "Interpolate", icon: Blend, hint: "RIFE" },
};

/** Node-style chain in the only valid order. Toggle with the switch, select to inspect. */
export function PipelineChain() {
  const stages = useApp((s) => s.stages);
  const selectedStage = useApp((s) => s.selectedStage);
  const selectStage = useApp((s) => s.selectStage);
  const toggleStage = useApp((s) => s.toggleStage);

  return (
    <ol className="flex flex-col" aria-label="Processing pipeline">
      {STAGE_ORDER.map((kind, i) => {
        const s = stages[kind];
        const meta = STAGE_META[kind];
                const soon = COMING_SOON.has(kind);
        const Icon = meta.icon;
        const detail =
          kind === "upscale" ? `×${s.scale}` : kind === "interpolate" ? `×${s.factor}` : kind === "denoise" ? `L${s.strength}` : "";
        return (
          <Fragment key={kind}>
            {i > 0 && (
              <li aria-hidden className="ml-[19px] h-2 w-px bg-gradient-to-b from-border-strong to-border" />
            )}
            <li>
              <div
                className={cn(
                  "group flex items-center gap-2 rounded-sm px-1.5 py-1.5 ring-1 transition-colors",
                  selectedStage === kind ? "bg-accent/10 ring-accent/40" : "ring-transparent hover:bg-white/4",
                  soon && "opacity-60",
                )}
              >
                <button
                  type="button"
                  onClick={() => selectStage(kind)}
                  className="flex min-w-0 flex-1 items-center gap-2 text-left focus-visible:outline-none"
                  aria-current={selectedStage === kind}
                >
                  <span
                    className={cn(
                      "grid size-6 shrink-0 place-items-center rounded-[5px] ring-1",
                      s.enabled ? "bg-accent/15 text-accent ring-accent/35" : "bg-black/25 text-fg-subtle ring-border",
                    )}
                  >
                    <Icon className="size-3.5" strokeWidth={1.75} />
                  </span>
                  <span className="min-w-0">
                    <span className={cn("block truncate text-[13px]", s.enabled ? "text-fg" : "text-fg-muted")}>{meta.label}</span>
                    <span className="block truncate font-mono text-[10px] text-fg-subtle">{meta.hint}</span>
                  </span>
                </button>
                {detail && s.enabled && <span className="tabular font-mono text-[11px] text-fg-muted">{detail}</span>}
                {COMING_SOON.has(kind) ? (
                  <Tooltip label="This stage's model runtime ships in an upcoming update.">
                    <span className="rounded-[4px] bg-white/6 px-1.5 py-0.5 text-[9px] font-semibold tracking-wider text-fg-subtle uppercase ring-1 ring-border">
                      Soon
                    </span>
                  </Tooltip>
                ) : (
                  <button
                    type="button"
                    role="switch"
                    aria-checked={s.enabled}
                    aria-label={`${meta.label} ${s.enabled ? "on" : "off"}`}
                    onClick={() => {
                      toggleStage(kind);
                      selectStage(kind);
                    }}
                    className={cn(
                      "relative h-4 w-7 shrink-0 rounded-full transition-colors focus-visible:outline-2 focus-visible:outline-accent",
                      s.enabled ? "bg-accent" : "bg-white/15",
                    )}
                  >
                    <span className={cn("absolute top-0.5 size-3 rounded-full bg-white shadow transition-[left]", s.enabled ? "left-[14px]" : "left-0.5")} />
                  </button>
                )}
              </div>
            </li>
          </Fragment>
        );
      })}
    </ol>
  );
}
