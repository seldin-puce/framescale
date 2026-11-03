import { Coffee, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { backend } from "@/ipc";
import { LINKS } from "@/lib/links";
import { useApp } from "@/stores/app";

/**
 * Shown right after a render finishes (first render, then rarely). Quiet,
 * dismissible, never blocking: a tip jar, not a paywall.
 */
export function SupportNudge() {
  const show = useApp((s) => s.nudge);
  const dismiss = useApp((s) => s.dismissNudge);
  const openAbout = useApp((s) => s.setAboutOpen);
  return (
    <div className="pointer-events-none fixed right-4 bottom-10 z-40">
      <AnimatePresence>
        {show && (
          <motion.div
            role="status"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="pointer-events-auto flex w-72 items-start gap-2.5 rounded-md border border-border-strong bg-surface-raised p-3 shadow-xl shadow-black/50"
          >
            <Coffee className="mt-0.5 size-4 shrink-0 text-[#db61a2]" />
            <div className="min-w-0 flex-1">
              <p className="text-xs font-medium text-fg">Render finished. Enjoying FrameScale?</p>
              <p className="mt-0.5 text-[11px] leading-relaxed text-fg-muted">It's free and open source. Tips help fund GPU testing.</p>
              <div className="mt-2 flex items-center gap-3">
                {LINKS.kofi && (
                  <button
                    type="button"
                    onClick={() => {
                      dismiss();
                      void backend.openExternal(LINKS.kofi);
                    }}
                    className="rounded-sm bg-[#ff5e5b] px-2 py-1 text-[11px] font-semibold text-white hover:brightness-110"
                  >
                    Buy me a coffee
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    dismiss();
                    openAbout(true);
                  }}
                  className="text-[11px] font-semibold text-fg-muted hover:text-fg"
                >
                  About
                </button>
              </div>
            </div>
            <button type="button" aria-label="Dismiss" onClick={dismiss} className="text-fg-subtle hover:text-fg">
              <X className="size-3.5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
