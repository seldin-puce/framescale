import { LoaderCircle, Pause, Play } from "lucide-react";
import { useRef } from "react";
import { timecode } from "@/lib/format";
import { IconButton } from "../ui/IconButton";

interface Props {
  frames: number;
  fps: number;
  frame: number;
  onSeek: (frame: number) => void;
  playing: boolean;
  onTogglePlay: () => void;
  buffering: boolean;
}

/** Full-length transport for comparing a finished render with its source. */
export function CompareTimeline({ frames, fps, frame, onSeek, playing, onTogglePlay, buffering }: Props) {
  const track = useRef<HTMLDivElement>(null);
  const last = Math.max(1, frames - 1);
  const seek = (clientX: number) => {
    const r = track.current?.getBoundingClientRect();
    if (r) onSeek(Math.round(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * last));
  };
  const pct = `${(frame / last) * 100}%`;
  return (
    <div className="flex h-12 items-center gap-2 rounded-md bg-surface/70 px-2 ring-1 ring-border">
      <IconButton icon={playing ? Pause : Play} label={playing ? "Pause (Space)" : "Play (Space)"} onClick={onTogglePlay} disabled={frames === 0} />
      <div
        ref={track}
        role="slider"
        tabIndex={0}
        aria-label="Playback position"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={frame}
        aria-valuetext={timecode(frame / fps)}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          seek(e.clientX);
        }}
        onPointerMove={(e) => e.buttons === 1 && seek(e.clientX)}
        onKeyDown={(e) => {
          const step = e.shiftKey ? Math.round(fps * 5) : 1;
          if (e.key === "ArrowLeft") onSeek(Math.max(0, frame - step));
          else if (e.key === "ArrowRight") onSeek(Math.min(last, frame + step));
          else if (e.key === " ") {
            e.preventDefault();
            onTogglePlay();
          } else return;
          e.preventDefault();
        }}
        className="relative h-7 flex-1 cursor-pointer rounded-sm focus-visible:outline-2 focus-visible:outline-accent"
      >
        <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded-full bg-white/10" />
        <div className="absolute top-1/2 h-1 -translate-y-1/2 rounded-full bg-accent" style={{ width: pct }} />
        <div className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white shadow ring-2 ring-accent" style={{ left: pct }} />
      </div>
      {buffering && <LoaderCircle className="size-3.5 animate-spin text-fg-muted" aria-label="Buffering" />}
      <span className="tabular w-40 text-right font-mono text-xs text-fg-muted">
        {timecode(frame / fps)} <span className="text-fg-subtle">/ {timecode(frames / fps)}</span>
      </span>
    </div>
  );
}
