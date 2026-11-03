import { CircleCheck, Coffee, ExternalLink, FolderOpen, Play, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { backend } from "@/ipc";
import { basename, prettyDir } from "@/lib/format";
import { LINKS } from "@/lib/links";
import { useApp } from "@/stores/app";

/**
 * Shown when a render finishes: what was made, where it is, and the next
 * step one click away: play it here against the original, show it in the file
 * manager, or open it in the system player. Stays until dismissed; the status
 * bar keeps a link afterwards.
 * When due, it carries one quiet "buy me a coffee" line (never a separate popup).
 */
export function CompletionCard() {
  const id = useApp((s) => s.completed);
  const dismissed = useApp((s) => s.completedDismissed);
  const job = useApp((s) => (s.completed ? s.jobs[s.completed] : undefined));
  const nudge = useApp((s) => s.nudge);
  const { dismissCompleted, dismissNudge, openOutput, revealOutput, playCompare } = useApp.getState();
  const show = !!id && !!job && job.state === "done" && !dismissed;
  const out = job?.spec.output.path ?? "";

  const btn =
    "flex items-center gap-1.5 rounded-sm px-2.5 py-1.5 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-accent";

  return (
    <div className="pointer-events-none fixed right-4 bottom-10 z-40">
      <AnimatePresence>
        {show && id && (
          <motion.div
            role="status"
            aria-live="polite"
            initial={{ opacity: 0, y: 12, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12 }}
            transition={{ duration: 0.2, ease: [0.1, 0.9, 0.2, 1] }}
            className="pointer-events-auto w-[340px] rounded-lg border border-border-strong bg-surface-raised shadow-2xl shadow-black/60"
          >
            <div className="flex items-start gap-2.5 p-3.5">
              <span className="grid size-8 shrink-0 place-items-center rounded-full bg-ok/15 text-ok ring-1 ring-ok/30">
                <CircleCheck className="size-4.5" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-[13px] font-semibold text-fg">Render finished</p>
                <p className="mt-0.5 truncate font-mono text-[11px] text-fg" title={out}>
                  {basename(out)}
                </p>
                <p className="truncate text-[11px] text-fg-subtle" title={out}>
                  in {prettyDir(out)}
                </p>
              </div>
              <button type="button" aria-label="Dismiss" onClick={dismissCompleted} className="rounded-sm p-0.5 text-fg-subtle hover:bg-white/6 hover:text-fg">
                <X className="size-3.5" />
              </button>
            </div>
            <div className="flex gap-1.5 px-3.5 pb-3.5">
              <button
                type="button"
                onClick={() => playCompare(id)}
                title="Play here, next to the original"
                className={`${btn} flex-1 justify-center bg-accent text-white hover:bg-accent-hover`}
              >
                <Play className="size-3.5" /> Play &amp; compare
              </button>
              <button type="button" onClick={() => void revealOutput(id)} className={`${btn} flex-1 justify-center bg-white/6 text-fg ring-1 ring-border-strong hover:bg-white/10`}>
                <FolderOpen className="size-3.5" /> Show in folder
              </button>
              <button
                type="button"
                onClick={() => void openOutput(id)}
                aria-label="Open in your video player"
                title="Open in your video player"
                className={`${btn} bg-white/6 text-fg ring-1 ring-border-strong hover:bg-white/10`}
              >
                <ExternalLink className="size-3.5" />
              </button>
            </div>
            {nudge && LINKS.kofi && (
              <div className="flex items-center justify-between border-t border-border px-3.5 py-2 text-[11px] text-fg-muted">
                <span>Enjoying FrameScale? It's free and open source.</span>
                <button
                  type="button"
                  onClick={() => {
                    dismissNudge();
                    void backend.openExternal(LINKS.kofi);
                  }}
                  className="flex shrink-0 items-center gap-1 font-semibold text-[#ff5e5b] hover:brightness-110"
                >
                  <Coffee className="size-3" /> Buy me a coffee
                </button>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
