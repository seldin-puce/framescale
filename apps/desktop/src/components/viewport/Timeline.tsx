import { Clapperboard, LoaderCircle, Pause, Play } from "lucide-react";
import { useRef } from "react";
import { cn } from "@/lib/cn";
import { timecode } from "@/lib/format";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";

export const SNIPPET_S = 2;

interface Props {
  duration: number;
  /** Start of the 2 s window, seconds. */
  t: number;
  onSeek: (t: number) => void;
  cuts: number[];
  canRender: boolean;
  renderHint: string;
  rendering: boolean;
  onRender: () => void;
  /** Playback of a rendered preview. */
  playable: boolean;
  playing: boolean;
  onTogglePlay: () => void;
  /** 0–1 progress through the preview while playing. */
  playhead: number | null;
}

/** Scrubber with a 2-second preview window, scene-cut markers and render/play controls. */
export function Timeline({ duration, t, onSeek, cuts, canRender, renderHint, rendering, onRender, playable, playing, onTogglePlay, playhead }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const dur = Math.max(duration, SNIPPET_S);
  const clamp = (x: number) => Math.max(0, Math.min(dur - SNIPPET_S, x));
  const pos = (x: number) => `${(x / dur) * 100}%`;
  const disabled = !canRender && !playable;

  const seek = (clientX: number) => {
    const r = track.current?.getBoundingClientRect();
    if (r) onSeek(clamp(((clientX - r.left) / r.width) * dur));
  };

  return (
    <div className="flex h-12 items-center gap-2 rounded-md bg-surface/70 px-2 ring-1 ring-border">
      <IconButton icon={playing ? Pause : Play} label={playable ? (playing ? "Pause preview" : "Play preview") : "Render a preview to play it"} onClick={onTogglePlay} disabled={!playable} />
      <div
        ref={track}
        role="slider"
        tabIndex={disabled ? -1 : 0}
        aria-label="Preview position"
        aria-valuemin={0}
        aria-valuemax={Math.round(dur)}
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
          if (e.key === "ArrowLeft") onSeek(clamp(t - step));
          if (e.key === "ArrowRight") onSeek(clamp(t + step));
        }}
        className={cn("relative h-7 flex-1 cursor-pointer rounded-sm focus-visible:outline-2 focus-visible:outline-accent", disabled && "cursor-not-allowed opacity-50")}
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10" />
        <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent/40" style={{ width: pos(t) }} />
        {cuts.map((c) => (
          <div key={c} className="absolute top-1 bottom-1 w-px bg-warn/70" style={{ left: pos(c) }} title={`Scene cut ${timecode(c)}`} />
        ))}
        <div className="absolute top-0.5 bottom-0.5 overflow-hidden rounded-[3px] border border-accent bg-accent/15" style={{ left: pos(t), width: pos(SNIPPET_S) }}>
          {playhead != null && <div className="absolute inset-y-0 left-0 bg-accent/35" style={{ width: `${playhead * 100}%` }} />}
        </div>
        <div className="absolute top-0 bottom-0 w-0.5 -translate-x-1/2 rounded-full bg-white shadow" style={{ left: pos(t) }} />
      </div>
      <span className="tabular w-36 text-right font-mono text-xs text-fg-muted">
        {timecode(t)} <span className="text-fg-subtle">/ {timecode(duration)}</span>
      </span>
      <Tooltip label={renderHint}>
        <span>
          <button
            type="button"
            disabled={!canRender || rendering}
            onClick={onRender}
            className="flex items-center gap-1.5 rounded-sm bg-accent px-2.5 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
          >
            {rendering ? <LoaderCircle className="size-3.5 animate-spin" /> : <Clapperboard className="size-3.5" />}
            {rendering ? "Rendering…" : "Render 2 s"}
          </button>
        </span>
      </Tooltip>
    </div>
  );
}
