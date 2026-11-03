import { CircleCheck } from "lucide-react";
import { basename, duration, fps } from "@/lib/format";
import { useApp } from "@/stores/app";

const CODEC_LABEL = { hevc10: "HEVC 10-bit", av1: "AV1 10-bit", h264: "H.264", proRes422Hq: "ProRes 422 HQ", proRes4444: "ProRes 4444" } as const;

export function StatusBar() {
  const jobs = useApp((s) => s.jobs);
  const progress = useApp((s) => s.progress);
  const output = useApp((s) => s.output);
  const active = Object.values(jobs).find((j) => ["analyzing", "rendering", "paused", "concatenating", "muxing"].includes(j.state));
  const p = active ? progress[active.id] : undefined;
  const lastId = useApp((s) => s.completed);
  const last = useApp((s) => (s.completed ? s.jobs[s.completed] : undefined));

  return (
    <footer className="flex h-7 shrink-0 items-center gap-3 border-t border-border bg-surface-glass px-3 text-[11px] text-fg-muted" role="status" aria-live="polite">
      {active ? (
        <>
          <span className="truncate text-fg">{active.state === "paused" ? "Paused" : "Rendering"} {basename(active.spec.input)}</span>
          {p && (
            <span className="tabular font-mono">
              segment {p.segment + 1}/{p.segmentsTotal} · {fps(p.fps)} fps · ETA {duration(p.etaS)}
            </span>
          )}
        </>
      ) : last?.state === "done" && lastId ? (
        <span className="flex min-w-0 items-center gap-1.5">
          <CircleCheck className="size-3 shrink-0 text-ok" />
          <span className="truncate">Finished {basename(last.spec.output.path)}</span>
          <button type="button" onClick={() => void useApp.getState().revealOutput(lastId)} className="shrink-0 text-accent hover:text-accent-hover">
            Show in folder
          </button>
        </span>
      ) : (
        <span>Idle</span>
      )}
      <span className="ml-auto font-mono">{CODEC_LABEL[output.codec]} · {output.container.toUpperCase()} · color passthrough</span>
    </footer>
  );
}
