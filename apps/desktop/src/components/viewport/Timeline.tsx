import { Clapperboard, Pause, Play } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "@/lib/cn";
import { timecode } from "@/lib/format";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";

const SNIPPET_S = 2;

/** Scrubber with a 2-second preview window and scene-cut markers. */
export function Timeline({ duration, cuts, disabled }: { duration: number; cuts: number[]; disabled: boolean }) {
  const [t, setT] = useState(duration * 0.3);
  const [playing, setPlaying] = useState(false);
  const track = useRef<HTMLDivElement>(null);
  const clamp = (x: number) => Math.max(0, Math.min(Math.max(0, duration - SNIPPET_S), x));
  const pos = (x: number) => `${(x / duration) * 100}%`;

  const seek = (clientX: number) => {
    const r = track.current?.getBoundingClientRect();
    if (!r) return;
    setT(clamp(((clientX - r.left) / r.width) * duration));
  };

  return (
    <div className="flex h-12 items-center gap-2 rounded-md bg-surface/70 px-2 ring-1 ring-border">
      <IconButton icon={playing ? Pause : Play} label={playing ? "Pause preview" : "Play preview"} onClick={() => setPlaying((p) => !p)} disabled={disabled} />
      <div
        ref={track}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label="Preview position"
        aria-valuemin={0}
        aria-valuemax={Math.round(duration)}
        aria-valuenow={Math.round(t)}
        aria-valuetext={timecode(t)}
        aria-disabled={disabled}
        onPointerDown={(e) => {
          if (disabled) return;
          e.currentTarget.setPointerCapture(e.pointerId);
          seek(e.clientX);
        }}
        onPointerMove={(e) => e.buttons === 1 && !disabled && seek(e.clientX)}
        onKeyDown={(e) => {
          const step = e.shiftKey ? 10 : 1;
          if (e.key === "ArrowLeft") setT((x) => clamp(x - step));
          if (e.key === "ArrowRight") setT((x) => clamp(x + step));
        }}
        className={cn("relative h-7 flex-1 cursor-pointer rounded-sm focus-visible:outline-2 focus-visible:outline-accent", disabled && "cursor-not-allowed opacity-50")}
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10" />
        <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent/40" style={{ width: pos(t) }} />
        {cuts.map((c) => (
          <div key={c} className="absolute top-1 bottom-1 w-px bg-warn/70" style={{ left: pos(c) }} title={`Scene cut ${timecode(c)}`} />
        ))}
        {/* 2 s preview window */}
        <div
          className="absolute top-0.5 bottom-0.5 rounded-[3px] border border-accent bg-accent/15"
          style={{ left: pos(t), width: pos(SNIPPET_S) }}
        />
        <div className="absolute top-0 bottom-0 w-0.5 -translate-x-1/2 rounded-full bg-white shadow" style={{ left: pos(t) }} />
      </div>
      <span className="tabular w-36 text-right font-mono text-xs text-fg-muted">
        {timecode(t)} <span className="text-fg-subtle">/ {timecode(duration)}</span>
      </span>
      <Tooltip label="Rendering a 2 s preview runs your exact pipeline on the GPU. Available once the ncnn inference backend lands (next step).">
        <span>
          <button
            type="button"
            disabled
            className="flex items-center gap-1.5 rounded-sm bg-accent px-2.5 py-1.5 text-xs font-semibold text-white disabled:opacity-40"
          >
            <Clapperboard className="size-3.5" /> Render 2 s
          </button>
        </span>
      </Tooltip>
    </div>
  );
}
