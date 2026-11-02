import { CircleX, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";
import { useApp } from "@/stores/app";

export function Toast() {
  const toast = useApp((s) => s.toast);
  useEffect(() => {
    if (!toast) return;
    const h = setTimeout(() => useApp.setState({ toast: null }), 6000);
    return () => clearTimeout(h);
  }, [toast]);
  return (
    <div className="pointer-events-none fixed bottom-10 left-1/2 z-50 -translate-x-1/2" aria-live="assertive">
      <AnimatePresence>
        {toast && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 8 }}
            className="pointer-events-auto flex max-w-lg items-start gap-2 rounded-md border border-err/30 bg-surface-raised px-3 py-2 text-xs text-fg shadow-xl shadow-black/50"
          >
            <CircleX className="mt-0.5 size-3.5 shrink-0 text-err" />
            <span className="select-text">{toast.message}</span>
            <button type="button" aria-label="Dismiss" onClick={() => useApp.setState({ toast: null })} className="text-fg-subtle hover:text-fg">
              <X className="size-3.5" />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
