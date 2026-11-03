import type { StageKind } from "@framescale/ipc-types";
import { ExternalLink } from "lucide-react";
import { backend } from "@/ipc";
import { LINKS } from "@/lib/links";
import { COMING_SOON, GENERAL_V3, MODELS, useApp, type StageState } from "@/stores/app";
import { Segmented } from "../ui/Segmented";
import { STAGE_META } from "./PipelineChain";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] text-fg-muted">{label}</span>
      {children}
    </label>
  );
}

const selectCls =
  "w-full rounded-sm border border-border-strong bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent";

export function StageInspector() {
  const kind = useApp((s) => s.selectedStage);
  const s = useApp((st) => st.stages[kind]);
  const setStage = useApp((st) => st.setStage);
  const set = (patch: Partial<StageState>) => setStage(kind, patch);

  return (
    <div className="flex flex-col gap-2.5">
      <div className="text-[13px] font-medium">{STAGE_META[kind].label}</div>
      {COMING_SOON.has(kind) && (
        <p className="rounded-sm bg-white/4 px-2 py-1.5 text-[11px] leading-relaxed text-fg-muted ring-1 ring-border">
          Coming soon: the engine currently runs Real-CUGAN denoising, Real-ESRGAN upscaling and RIFE interpolation.
        </p>
      )}
      {kind !== "deinterlace" && kind !== "faceFix" && !COMING_SOON.has(kind) && (
        <Field label="Model">
          <select className={selectCls} value={s.model} onChange={(e) => set({ model: e.target.value })}>
            {MODELS[kind as StageKind].map((m) => (
              <option key={m.id} value={m.id}>{m.label}</option>
            ))}
          </select>
        </Field>
      )}
      {kind === "upscale" && (
        <Field label="Scale">
          <Segmented ariaLabel="Upscale factor" value={String(s.scale)} onChange={(v) => set({ scale: Number(v) as 2 | 4 })}
            options={[{ value: "2", label: "2×" }, { value: "4", label: "4×" }]} />
        </Field>
      )}
      {kind === "upscale" && s.model === GENERAL_V3 && (
        <Field label={`Noise removal · ${Math.round(s.denoise * 100)}%`}>
          <input type="range" min={0} max={1} step={0.05} value={s.denoise} onChange={(e) => set({ denoise: Number(e.target.value) })}
            className="accent-accent" aria-describedby="general-denoise-help" />
          <span id="general-denoise-help" className="text-[10px] leading-snug text-fg-subtle">
            Lower keeps film grain and fine texture; higher cleans noise but smooths more. 50% is Real-ESRGAN's default.
          </span>
        </Field>
      )}
      {kind === "denoise" && (
        <Field label="Strength">
          {/* Real-CUGAN variants: SE has conservative + denoise 1x/2x/3x; Pro has conservative + 3x. */}
          {s.model === "realcugan-pro" ? (
            <Segmented ariaLabel="Denoise strength" value={s.strength >= 2 ? "3" : "0"} onChange={(v) => set({ strength: Number(v) as 0 | 3 })}
              options={[{ value: "0", label: "Gentle" }, { value: "3", label: "Strong" }]} />
          ) : (
            <Segmented ariaLabel="Denoise strength" value={String(s.strength)} onChange={(v) => set({ strength: Number(v) as 0 | 1 | 2 | 3 })}
              options={[{ value: "0", label: "Gentle" }, { value: "1", label: "Low" }, { value: "2", label: "Medium" }, { value: "3", label: "Strong" }]} />
          )}
          <span className="text-[10px] leading-snug text-fg-subtle">
            Gentle restores without smoothing grain. Real-CUGAN is trained on anime and line art; on live action, start Gentle or Low.
          </span>
        </Field>
      )}
      {kind === "deinterlace" && (
        <Field label="Output">
          <Segmented ariaLabel="Deinterlace output" value={s.sendField ? "field" : "frame"} onChange={(v) => set({ sendField: v === "field" })}
            options={[{ value: "field", label: "Double rate" }, { value: "frame", label: "Same rate" }]} />
        </Field>
      )}
      {kind === "interpolate" && (
        <>
          <Field label="Frame-rate multiplier">
            <Segmented ariaLabel="Interpolation factor" value={String(s.factor)} onChange={(v) => set({ factor: Number(v) as 2 | 3 | 4 })}
              options={[{ value: "2", label: "2×" }, { value: "3", label: "3×" }, { value: "4", label: "4×" }]} />
          </Field>
          <Field label={`Scene-cut sensitivity · ${s.sceneThreshold}`}>
            <input type="range" min={4} max={30} value={s.sceneThreshold} onChange={(e) => set({ sceneThreshold: Number(e.target.value) })}
              className="accent-accent" aria-describedby="scene-help" />
            <span id="scene-help" className="text-[10px] leading-snug text-fg-subtle">
              Frames are held, not morphed, across detected hard cuts. Lower values catch more cuts.
            </span>
          </Field>
        </>
      )}
      {kind === "faceFix" && (
        <>
          <Field label={`Fidelity · ${s.fidelity.toFixed(2)}`}>
            <input type="range" min={0} max={1} step={0.05} value={s.fidelity} onChange={(e) => set({ fidelity: Number(e.target.value) })}
              className="accent-accent" aria-describedby="fidelity-help" />
            <span id="fidelity-help" className="text-[10px] leading-snug text-fg-subtle">
              Lower restores more strongly; higher stays closer to the original face. Faces are found automatically (YuNet).
            </span>
          </Field>
          <p className="text-[10px] leading-snug text-fg-subtle">
            CodeFormer is © S-Lab, NTU, licensed for non-commercial use only (S-Lab License 1.0).{" "}
            <button type="button" onClick={() => void backend.openExternal(LINKS.codeformer)} className="inline-flex items-center gap-0.5 text-accent hover:text-accent-hover">
              About CodeFormer <ExternalLink className="size-2.5" />
            </button>
          </p>
        </>
      )}
    </div>
  );
}
