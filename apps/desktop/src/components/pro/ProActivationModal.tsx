import * as Dialog from "@radix-ui/react-dialog";
import type { Feature } from "@framescale/ipc-types";
import { Check, KeyRound, Sparkles, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";
import { cn } from "@/lib/cn";
import { useApp } from "@/stores/app";

const FEATURE_COPY: Record<Feature, string> = {
  faceFix: "Face restoration with your own CodeFormer / GFPGAN model",
  interpolateHighFactor: "3× and 4× frame interpolation",
  proResOutput: "Apple ProRes 422 HQ / 4444 export",
  outputAbove4k: "Output above 4K (up to 8K)",
  batchReorder: "Drag-to-reorder batch queue",
  batchUnlimited: "Unlimited batch size",
  multiGpu: "Split renders across multiple GPUs",
  byom: "Bring-your-own models",
};

const PERKS: Feature[] = ["batchUnlimited", "faceFix", "interpolateHighFactor", "proResOutput", "outputAbove4k", "multiGpu"];

export function ProActivationModal() {
  const { open, feature } = useApp((s) => s.proModal);
  const close = useApp((s) => s.closePro);
  const activate = useApp((s) => s.activate);
  const tier = useApp((s) => s.entitlements.tier);
  const [token, setToken] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const err = await activate(token);
    setBusy(false);
    if (err) setError(err);
    else {
      setToken("");
      close();
    }
  };

  return (
    <Dialog.Root open={open} onOpenChange={(o) => !o && close()}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div
                className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
              />
            </Dialog.Overlay>
            <Dialog.Content asChild forceMount aria-describedby={undefined}>
              <motion.div
                className="fixed top-1/2 left-1/2 z-50 w-[440px] max-w-[calc(100vw-32px)] rounded-lg border border-border-strong bg-surface p-5 shadow-2xl shadow-black/60"
                initial={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                animate={{ opacity: 1, scale: 1, x: "-50%", y: "-50%" }}
                exit={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                transition={{ duration: 0.18, ease: [0.1, 0.9, 0.2, 1] }}
              >
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="grid size-9 place-items-center rounded-md bg-pro/15 text-pro ring-1 ring-pro/30">
                      <Sparkles className="size-4.5" />
                    </span>
                    <div>
                      <Dialog.Title className="text-[15px] font-semibold">FrameScale Pro</Dialog.Title>
                      <p className="text-xs text-fg-muted">
                        {feature ? FEATURE_COPY[feature] : "Unlock the full studio pipeline"}
                      </p>
                    </div>
                  </div>
                  <Dialog.Close className="rounded-sm p-1 text-fg-muted hover:bg-white/6 hover:text-fg" aria-label="Close">
                    <X className="size-4" />
                  </Dialog.Close>
                </div>

                <ul className="mb-4 grid grid-cols-2 gap-x-3 gap-y-1.5">
                  {PERKS.map((f) => (
                    <li key={f} className={cn("flex items-start gap-1.5 text-xs text-fg-muted", f === feature && "text-fg")}>
                      <Check className="mt-0.5 size-3 shrink-0 text-ok" strokeWidth={2.5} />
                      {FEATURE_COPY[f]}
                    </li>
                  ))}
                </ul>

                {tier === "pro" ? (
                  <p className="rounded-sm bg-ok/10 px-3 py-2 text-xs text-ok ring-1 ring-ok/25">Pro is active on this device.</p>
                ) : (
                  <form
                    onSubmit={(e) => {
                      e.preventDefault();
                      void submit();
                    }}
                  >
                    <label htmlFor="license" className="mb-1 flex items-center gap-1.5 text-xs font-medium text-fg-muted">
                      <KeyRound className="size-3.5" /> License key
                    </label>
                    <textarea
                      id="license"
                      value={token}
                      onChange={(e) => setToken(e.target.value)}
                      rows={2}
                      spellCheck={false}
                      placeholder="Paste the key from your purchase email"
                      className="w-full resize-none rounded-sm border border-border-strong bg-black/30 px-2.5 py-2 font-mono text-[11px] text-fg outline-none select-text placeholder:text-fg-subtle focus:border-accent"
                    />
                    {error && <p role="alert" className="mt-1.5 text-xs text-err">{error}</p>}
                    <div className="mt-3 flex items-center justify-between">
                      <a href="https://framescale.ai/pro" target="_blank" rel="noreferrer" className="text-xs text-accent hover:text-accent-hover">
                        Get Pro →
                      </a>
                      <button
                        type="submit"
                        disabled={!token.trim() || busy}
                        className="rounded-sm bg-pro px-3 py-1.5 text-xs font-semibold text-black transition-opacity hover:opacity-90 disabled:opacity-40"
                      >
                        {busy ? "Verifying…" : "Activate"}
                      </button>
                    </div>
                    <p className="mt-3 text-[10px] leading-relaxed text-fg-subtle">
                      Keys are verified offline with an Ed25519 signature; nothing about your media leaves this machine.
                    </p>
                  </form>
                )}
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
