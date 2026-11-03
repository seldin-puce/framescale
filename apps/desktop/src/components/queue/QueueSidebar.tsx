import { DndContext, KeyboardSensor, PointerSensor, closestCenter, useSensor, useSensors, type DragEndEvent } from "@dnd-kit/core";
import { SortableContext, arrayMove, sortableKeyboardCoordinates, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { FolderOpen, Sparkles, Upload } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { backend } from "@/ipc";
import { cn } from "@/lib/cn";
import { useApp } from "@/stores/app";
import { DraftRow, JobRow } from "./QueueItem";

function DropZone({ hover }: { hover: boolean }) {
  const add = useApp((s) => s.addFiles);
  return (
    <button
      type="button"
      onClick={() => void backend.pickFiles().then(add)}
      className={cn(
        "flex w-full flex-col items-center gap-1.5 rounded-md border border-dashed px-3 py-4 text-center transition-colors focus-visible:outline-2 focus-visible:outline-accent",
        hover ? "border-accent bg-accent/10 text-accent" : "border-border-strong text-fg-muted hover:border-fg-subtle hover:bg-white/3",
      )}
    >
      <Upload className="size-5" strokeWidth={1.5} />
      <span className="text-xs font-medium">{hover ? "Drop to add" : "Drop videos here"}</span>
      <span className="flex items-center gap-1 text-[11px] text-fg-subtle">
        <FolderOpen className="size-3" /> or browse
      </span>
    </button>
  );
}

export function QueueSidebar() {
  const drafts = useApp((s) => s.drafts);
  const jobsMap = useApp((s) => s.jobs);
  const progress = useApp((s) => s.progress);
  const renderAll = useApp((s) => s.renderAll);
  const addFiles = useApp((s) => s.addFiles);
  const fail = useApp((s) => s.fail);
  const [hover, setHover] = useState(false);

  useEffect(() => {
    let off: (() => void) | undefined;
    void backend.onFileDrop((p) => void addFiles(p), setHover).then((u) => (off = u));
    return () => off?.();
  }, [addFiles]);

  const jobs = useMemo(() => Object.values(jobsMap).sort((a, b) => a.queuePos - b.queuePos), [jobsMap]);
  const [order, setOrder] = useState<string[] | null>(null);
  const shown = order ? order.map((id) => jobsMap[id]).filter((j) => j != null) : jobs;
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }), useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates }));

  const onDragEnd = async (e: DragEndEvent) => {
    if (!e.over || e.active.id === e.over.id) return;
    const ids = shown.map((j) => j.id as string);
    const next = arrayMove(ids, ids.indexOf(String(e.active.id)), ids.indexOf(String(e.over.id)));
    setOrder(next); // optimistic
    try {
      await backend.reorderJobs(next as never);
    } catch (err) {
      setOrder(null);
      fail(err);
    }
  };

  const ready = drafts.filter((d) => d.info).length;
  const active = jobs.filter((j) => !["done", "failed", "cancelled"].includes(j.state)).length;

  return (
    <aside className="flex h-full flex-col bg-surface-glass" aria-label="Queue">
      <div className="flex items-center justify-between px-3 pt-3 pb-2">
        <h2 className="text-[11px] font-semibold tracking-[0.08em] text-fg-subtle uppercase">Queue</h2>
        <span className="tabular font-mono text-[10px] text-fg-subtle">
          {active} active
        </span>
      </div>
      <div className="px-3">
        <DropZone hover={hover} />
      </div>

      <div className="mt-2 min-h-0 flex-1 overflow-y-auto px-1.5 pb-2">
        {drafts.length > 0 && (
          <>
            <div className="px-1.5 pt-2 pb-1 text-[10px] font-medium text-fg-subtle uppercase">Ready to render</div>
            <ul className="flex flex-col gap-0.5">
              {drafts.map((d) => (
                <DraftRow key={d.path} {...d} />
              ))}
            </ul>
          </>
        )}
        {shown.length > 0 && (
          <>
            <div className="flex items-center justify-between px-1.5 pt-3 pb-1 text-[10px] font-medium text-fg-subtle uppercase">
              <span>Jobs</span>
            </div>
            <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={(e) => void onDragEnd(e)}>
              <SortableContext items={shown.map((j) => j.id)} strategy={verticalListSortingStrategy}>
                <ul className="flex flex-col gap-0.5">
                  {shown.map((j) => (
                    <JobRow key={j.id} job={j} progress={progress[j.id]} sortable={j.state === "queued"} />
                  ))}
                </ul>
              </SortableContext>
            </DndContext>
          </>
        )}
        {drafts.length === 0 && shown.length === 0 && (
          <p className="px-3 pt-6 text-center text-xs leading-relaxed text-fg-subtle">
            Add clips, tune the pipeline on the right, then render. Everything runs locally on your GPU.
          </p>
        )}
      </div>

      <div className="border-t border-border p-3">
        <button
          type="button"
          disabled={ready === 0}
          onClick={() => void renderAll()}
          className="flex w-full items-center justify-center gap-1.5 rounded-sm bg-accent py-2 text-[13px] font-semibold text-white shadow-[0_1px_0_rgba(255,255,255,0.15)_inset] transition-colors hover:bg-accent-hover disabled:bg-white/8 disabled:text-fg-subtle"
        >
          <Sparkles className="size-3.5" />
          {ready > 1 ? `Render ${ready} videos` : "Render"}
        </button>
      </div>
    </aside>
  );
}
