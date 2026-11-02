import type { JobProgress, JobSummary, MediaInfo } from "@framescale/ipc-types";
import { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import { CircleCheck, CircleX, Film, GripVertical, LoaderCircle, Pause, Play, RotateCcw, Square, Trash2, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/cn";
import { basename, duration } from "@/lib/format";
import { useApp } from "@/stores/app";
import { IconButton } from "../ui/IconButton";

const STATE_LABEL: Record<JobSummary["state"], string> = {
  queued: "Queued",
  analyzing: "Analyzing",
  rendering: "Rendering",
  paused: "Paused",
  interrupted: "Interrupted",
  concatenating: "Joining",
  muxing: "Muxing",
  done: "Done",
  failed: "Failed",
  cancelled: "Cancelled",
};

function Thumb({ info, hdr }: { info: MediaInfo | null; hdr: boolean }) {
  return (
    <div className="relative grid h-10 w-16 shrink-0 place-items-center overflow-hidden rounded-[5px] bg-gradient-to-br from-surface-raised to-black ring-1 ring-border">
      <Film className="size-4 text-fg-subtle" strokeWidth={1.5} />
      {info && (
        <span className="absolute right-0.5 bottom-0.5 rounded-[3px] bg-black/70 px-1 font-mono text-[8px] leading-tight text-fg-muted">
          {info.video.height >= 2160 ? "4K" : `${info.video.height}p`}
        </span>
      )}
      {hdr && <span className="absolute top-0.5 left-0.5 rounded-[3px] bg-pro/90 px-1 text-[8px] font-bold leading-tight text-black">HDR</span>}
    </div>
  );
}

export function DraftRow({ path, info, error }: { path: string; info: MediaInfo | null; error: string | null }) {
  const selected = useApp((s) => s.selected === path);
  const select = useApp((s) => s.select);
  const remove = useApp((s) => s.removeDraft);
  const hdr = info?.video.color.transfer === "pq" || info?.video.color.transfer === "hlg";
  return (
    <li
      className={cn("group flex cursor-pointer items-center gap-2 rounded-sm p-1.5 ring-1", selected ? "bg-accent/10 ring-accent/40" : "ring-transparent hover:bg-white/4")}
      onClick={() => select(path)}
    >
      <Thumb info={info} hdr={hdr} />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px]" title={path}>{basename(path)}</div>
        <div className="truncate font-mono text-[10px] text-fg-subtle">
          {error ? <span className="text-err">{error}</span> : info ? `${info.video.width}×${info.video.height} · ${(info.video.avgFrameRate.num / info.video.avgFrameRate.den).toFixed(2)} fps · ${duration(info.durationS)}` : "Probing…"}
        </div>
      </div>
      <IconButton icon={Trash2} label="Remove" size="sm" className="opacity-0 group-hover:opacity-100 focus-visible:opacity-100" onClick={(e) => { e.stopPropagation(); remove(path); }} />
    </li>
  );
}

export function JobRow({ job, progress, sortable }: { job: JobSummary; progress: JobProgress | undefined; sortable: boolean }) {
  const selected = useApp((s) => s.selected === job.id);
  const select = useApp((s) => s.select);
  const act = useApp((s) => s.jobAction);
  const canReorder = useApp((s) => s.has("batchReorder"));
  const openPro = useApp((s) => s.openPro);
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({ id: job.id, disabled: !sortable || !canReorder });

  const total = job.framesTotal ?? progress?.framesTotal ?? 0;
  const done = progress?.framesDone ?? job.framesDone;
  const ratio = total > 0 ? Math.min(1, done / total) : 0;
  const s = job.state;
  const running = s === "rendering" || s === "analyzing" || s === "concatenating" || s === "muxing";

  const StatusIcon = s === "done" ? CircleCheck : s === "failed" ? CircleX : s === "interrupted" ? TriangleAlert : running ? LoaderCircle : null;

  return (
    <li
      ref={setNodeRef}
      style={{ transform: CSS.Transform.toString(transform), transition }}
      className={cn(
        "group flex cursor-pointer items-center gap-2 rounded-sm p-1.5 ring-1",
        selected ? "bg-accent/10 ring-accent/40" : "ring-transparent hover:bg-white/4",
        isDragging && "z-10 bg-surface-raised shadow-lg",
      )}
      onClick={() => select(job.id)}
    >
      {sortable && (
        canReorder ? (
          <button type="button" className="cursor-grab text-fg-subtle hover:text-fg" aria-label="Drag to reorder" {...attributes} {...listeners}>
            <GripVertical className="size-3.5" />
          </button>
        ) : (
          <button type="button" className="text-fg-subtle/60" aria-label="Reordering requires Pro" onClick={(e) => { e.stopPropagation(); openPro("batchReorder"); }}>
            <GripVertical className="size-3.5" />
          </button>
        )
      )}
      <Thumb info={null} hdr={false} />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1">
          <span className="truncate text-[13px]" title={job.spec.input}>{basename(job.spec.input)}</span>
        </div>
        <div className="mt-1 flex items-center gap-1.5">
          {StatusIcon && <StatusIcon className={cn("size-3 shrink-0", running && "animate-spin text-accent", s === "done" && "text-ok", s === "failed" && "text-err", s === "interrupted" && "text-warn")} />}
          <span className={cn("text-[10px]", s === "failed" ? "text-err" : "text-fg-muted")} title={job.error ?? undefined}>
            {STATE_LABEL[s]}
            {s === "rendering" && progress?.etaS != null ? ` · ${duration(progress.etaS)} left` : ""}
          </span>
          {(s === "rendering" || s === "paused") && total > 0 && (
            <span className="tabular ml-auto font-mono text-[10px] text-fg-subtle">{Math.round(ratio * 100)}%</span>
          )}
        </div>
        {(s === "rendering" || s === "paused" || s === "interrupted") && (
          <div className="mt-1 h-1 overflow-hidden rounded-full bg-white/8">
            <div className={cn("h-full rounded-full transition-[width] duration-500", s === "paused" || s === "interrupted" ? "bg-warn" : "bg-accent")} style={{ width: `${ratio * 100}%` }} />
          </div>
        )}
      </div>
      <div className="flex opacity-0 group-hover:opacity-100 focus-within:opacity-100" onClick={(e) => e.stopPropagation()}>
        {s === "rendering" && <IconButton icon={Pause} label="Pause (keeps GPU memory)" size="sm" onClick={() => void act(job.id, "pause")} />}
        {s === "rendering" && <IconButton icon={Square} label="Pause & free GPU (finishes current segment)" size="sm" onClick={() => void act(job.id, "hardPause")} />}
        {(s === "paused") && <IconButton icon={Play} label="Resume" size="sm" onClick={() => void act(job.id, "resume")} />}
        {(s === "interrupted" || s === "failed") && <IconButton icon={RotateCcw} label="Resume from last segment" size="sm" onClick={() => void act(job.id, "resume")} />}
        {!["done", "failed", "cancelled", "interrupted"].includes(s) && <IconButton icon={CircleX} label="Cancel" size="sm" onClick={() => void act(job.id, "cancel")} />}
        {["done", "failed", "cancelled", "interrupted"].includes(s) && <IconButton icon={Trash2} label="Remove from list" size="sm" onClick={() => void act(job.id, "remove")} />}
      </div>
    </li>
  );
}

