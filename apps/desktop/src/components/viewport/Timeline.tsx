import { Clapperboard, LoaderCircle, Pause, Play, X } from "lucide-react";
import { useRef } from "react";
import { cn } from "@/lib/cn";
import { timecode } from "@/lib/format";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";

export const SNIPPET_S = 2;

/** A preview render in flight, as the timeline shows it. */
export interface RenderRun {
  /** 0–1, or null while the amount of work isn't known yet. */
  fraction: number | null;
  /** Short status, e.g. "45% · 4 s left". */
  label: string;
  /** Longer status for the tooltip. */
  detail: string;
}

interface Props {
  duration: number;
  /** Start of the 2 s window, seconds. */
  t: number;
  onSeek: (t: number) => void;
  cuts: number[];
  /** A clip is shown, so the marker can move. */
  scrubbable: boolean;
  canRender: boolean;
  renderHint: string;
  /** The window is rendering: the marker stays put and the window fills up. */
  run: RenderRun | null;
  onRender: () => void;
  onCancel: () => void;
  /** Playback of a rendered preview. */
  playable: boolean;
  playing: boolean;
  onTogglePlay: () => void;
  /** 0–1 progress through the preview while playing. */
  playhead: number | null;
}

/** Scrubber with a 2-second preview window, scene-cut markers and render/play controls. */
export function Timeline(p: Props) {
  const { duration, t, onSeek, cuts, scrubbable, canRender, renderHint, run, onRender, onCancel, playable, playing, onTogglePlay, playhead } = p;
  const track = useRef<HTMLDivElement>(null);
  const dur = Math.max(duration, SNIPPET_S);
  const clamp = (x: number) => Math.max(0, Math.min(dur - SNIPPET_S, x));
  const pos = (x: number) => `${(x / dur) * 100}%`;
  const disabled = !scrubbable;
  const locked = run != null;
  const still = disabled || locked;

  const seek = (clientX: number) => {
    const r = track.current?.getBoundingClientRect();
    if (r) onSeek(clamp(((clientX - r.left) / r.width) * dur));
  };

  const slider = (
    <div
      ref={track}
      role="slider"
      tabIndex={disabled ? -1 : 0}
      aria-label="Preview position"
      aria-valuemin={0}
      aria-valuemax={Math.round(dur)}
      aria-valuenow={Math.round(t)}
      aria-valuetext={timecode(t)}
      aria-disabled={still}
      onPointerDown={(e) => {
        if (still) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        seek(e.clientX);
      }}
      onPointerMove={(e) => e.buttons === 1 && !still && seek(e.clientX)}
      onKeyDown={(e) => {
        if (still) return;
        const step = e.shiftKey ? 10 : 1;
        if (e.key === "ArrowLeft") onSeek(clamp(t - step));
        if (e.key === "ArrowRight") onSeek(clamp(t + step));
      }}
      className={cn(
        "relative h-7 w-full cursor-pointer rounded-sm focus-visible:outline-2 focus-visible:outline-accent",
        disabled && "cursor-not-allowed opacity-50",
        locked && "cursor-default",
      )}
    >
      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10" />
      <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent/40" style={{ width: pos(t) }} />
      {cuts.map((c) => (
        <div key={c} className="absolute top-1 bottom-1 w-px bg-warn/70" style={{ left: pos(c) }} title={`Scene cut ${timecode(c)}`} />
      ))}
      <div className="absolute top-0.5 bottom-0.5 overflow-hidden rounded-[3px] border border-accent bg-accent/15" style={{ left: pos(t), width: pos(SNIPPET_S) }}>
        {run ? (
          <div
            className={cn("absolute inset-y-0 left-0 bg-accent/45 transition-[width] duration-300", run.fraction == null && "w-full animate-pulse bg-accent/25")}
            style={run.fraction == null ? undefined : { width: `${run.fraction * 100}%` }}
          />
        ) : (
          playhead != null && <div className="absolute inset-y-0 left-0 bg-accent/35" style={{ width: `${playhead * 100}%` }} />
        )}
      </div>
      <div className="absolute top-0 bottom-0 w-0.5 -translate-x-1/2 rounded-full bg-white shadow" style={{ left: pos(t) }} />
    </div>
  );

  return (
    <div className="flex h-12 items-center gap-2 rounded-md bg-surface/70 px-2 ring-1 ring-border">
      <IconButton icon={playing ? Pause : Play} label={playable ? (playing ? "Pause preview" : "Play preview") : "Render a preview to play it"} onClick={onTogglePlay} disabled={!playable} />
      <div className="min-w-0 flex-1">
        {locked ? (
          <Tooltip label="Rendering this window. Cancel to move it.">
            {slider}
          </Tooltip>
        ) : (
          slider
        )}
      </div>
      <span className="tabular w-36 text-right font-mono text-xs text-fg-muted">
        {timecode(t)} <span className="text-fg-subtle">/ {timecode(duration)}</span>
      </span>
      {run ? (
        <div className="flex items-center gap-1">
          <Tooltip label={run.detail}>
            <div
              role="progressbar"
              aria-label="Preview render"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={run.fraction == null ? undefined : Math.round(run.fraction * 100)}
              aria-valuetext={run.label}
              className="relative flex h-7 w-40 items-center gap-1.5 overflow-hidden rounded-sm bg-accent/10 px-2 ring-1 ring-accent/40"
            >
              <div
                className={cn("absolute inset-y-0 left-0 bg-accent/30 transition-[width] duration-300", run.fraction == null && "w-full animate-pulse bg-accent/15")}
                style={run.fraction == null ? undefined : { width: `${run.fraction * 100}%` }}
              />
              <LoaderCircle className="relative size-3.5 shrink-0 animate-spin text-accent" />
              <span className="tabular relative truncate text-xs font-medium text-fg">{run.label}</span>
            </div>
          </Tooltip>
          <IconButton icon={X} label="Cancel the preview" size="sm" className="size-7" onClick={onCancel} />
        </div>
      ) : (
        <Tooltip label={renderHint}>
          <span>
            <button
              type="button"
              disabled={!canRender}
              onClick={onRender}
              className="flex items-center gap-1.5 rounded-sm bg-accent px-2.5 py-1.5 text-xs font-semibold text-white transition-colors hover:bg-accent-hover disabled:opacity-40"
            >
              <Clapperboard className="size-3.5" />
              Render 2 s
            </button>
          </span>
        </Tooltip>
      )}
    </div>
  );
}
