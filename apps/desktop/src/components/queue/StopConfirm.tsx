import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion } from "motion/react";
import { basename } from "@/lib/format";
import { useApp } from "@/stores/app";

/** Asked before a stop throws away more than a minute of rendering. */
export function StopConfirm() {
  const id = useApp((s) => s.confirmStop);
  const job = useApp((s) => (s.confirmStop ? s.jobs[s.confirmStop] : undefined));
  const progress = useApp((s) => (s.confirmStop ? s.progress[s.confirmStop] : undefined));
  const setConfirm = useApp((s) => s.setConfirmStop);
  const stopJob = useApp((s) => s.stopJob);

  const total = job?.framesTotal ?? progress?.framesTotal ?? 0;
  const done = progress?.framesDone ?? job?.framesDone ?? 0;
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;
  const open = id != null && job != null;

  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && setConfirm(null)}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
            </Dialog.Overlay>
            <Dialog.Content asChild forceMount>
              <motion.div
                className="fixed top-1/2 left-1/2 z-50 w-[380px] max-w-[calc(100vw-32px)] rounded-lg border border-border-strong bg-surface p-5 shadow-2xl shadow-black/60"
                initial={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                animate={{ opacity: 1, scale: 1, x: "-50%", y: "-50%" }}
                exit={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                transition={{ duration: 0.18, ease: [0.1, 0.9, 0.2, 1] }}
              >
                <Dialog.Title className="text-[15px] font-semibold">Stop this render?</Dialog.Title>
                <Dialog.Description className="mt-2 text-xs leading-relaxed text-fg-muted">
                  {pct != null ? `${pct}% of ` : ""}
                  <span className="text-fg">{basename(job.spec.input)}</span>
                  {pct != null ? " is done and" : ""} will be discarded. The clip goes back to Ready to render, so you can change the pipeline and render it
                  again. To keep the progress, pause instead.
                </Dialog.Description>
                <div className="mt-5 flex justify-end gap-2">
                  <Dialog.Close className="rounded-sm px-3 py-2 text-xs font-medium text-fg-muted hover:bg-white/6 hover:text-fg">Keep rendering</Dialog.Close>
                  <button
                    type="button"
                    autoFocus
                    onClick={() => id && void stopJob(id)}
                    className="rounded-sm bg-err px-3 py-2 text-xs font-semibold text-white hover:brightness-110"
                  >
                    Stop render
                  </button>
                </div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
