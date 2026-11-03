import type { StageKind } from "@framescale/ipc-types";
import { COMING_SOON, MODELS, useApp, type StageState } from "@/stores/app";
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
          Coming soon: the engine currently runs Real-ESRGAN upscaling and RIFE interpolation.
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
      {kind === "denoise" && (
        <Field label="Strength">
          <Segmented ariaLabel="Denoise strength" value={String(s.strength)} onChange={(v) => set({ strength: Number(v) as 0 | 1 | 2 | 3 })}
            options={[0, 1, 2, 3].map((n) => ({ value: String(n), label: n === 0 ? "Off" : `L${n}` }))} />
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
        <p className="text-[11px] leading-relaxed text-fg-muted">
          Face restoration runs <em>your</em> CodeFormer or GFPGAN weights (NCNN or ONNX). FrameScale doesn't ship encumbered models.
        </p>
      )}
    </div>
  );
}
