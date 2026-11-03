import * as Dialog from "@radix-ui/react-dialog";
import { CodeXml, Coffee, Heart, Scale, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { backend } from "@/ipc";
import { LINKS, hasDonationLinks } from "@/lib/links";
import { useApp } from "@/stores/app";

const VERSION = "0.1.0";

/** External link as a button: opened in the system browser via the opener plugin. */
function LinkButton({ href, icon: Icon, children, color }: { href: string; icon: typeof Heart; children: React.ReactNode; color: string }) {
  return (
    <button
      type="button"
      onClick={() => void backend.openExternal(href)}
      style={{ backgroundColor: color }}
      className="flex flex-1 items-center justify-center gap-1.5 rounded-sm px-3 py-2 text-xs font-semibold text-white transition-[filter] hover:brightness-110 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-white"
    >
      <Icon className="size-3.5" /> {children}
    </button>
  );
}

/** About + support. Free software: donations are optional and unlock nothing. */
export function AboutDialog() {
  const open = useApp((s) => s.aboutOpen);
  const setOpen = useApp((s) => s.setAboutOpen);

  return (
    <Dialog.Root open={open} onOpenChange={setOpen}>
      <AnimatePresence>
        {open && (
          <Dialog.Portal forceMount>
            <Dialog.Overlay asChild forceMount>
              <motion.div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} />
            </Dialog.Overlay>
            <Dialog.Content asChild forceMount aria-describedby={undefined}>
              <motion.div
                className="fixed top-1/2 left-1/2 z-50 w-[420px] max-w-[calc(100vw-32px)] rounded-lg border border-border-strong bg-surface p-5 shadow-2xl shadow-black/60"
                initial={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                animate={{ opacity: 1, scale: 1, x: "-50%", y: "-50%" }}
                exit={{ opacity: 0, scale: 0.97, x: "-50%", y: "-48%" }}
                transition={{ duration: 0.18, ease: [0.1, 0.9, 0.2, 1] }}
              >
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex items-center gap-2.5">
                    <span className="grid size-9 place-items-center rounded-md bg-gradient-to-br from-accent to-[#8fb8ff] text-sm font-black text-white">F</span>
                    <div>
                      <Dialog.Title className="text-[15px] font-semibold">FrameScale AI</Dialog.Title>
                      <p className="font-mono text-[11px] text-fg-muted">v{VERSION} · free &amp; open source</p>
                    </div>
                  </div>
                  <Dialog.Close className="rounded-sm p-1 text-fg-muted hover:bg-white/6 hover:text-fg" aria-label="Close">
                    <X className="size-4" />
                  </Dialog.Close>
                </div>

                <p className="text-xs leading-relaxed text-fg-muted">
                  Every feature is free, with no accounts, no watermarks and no limits. Your video never leaves this computer.
                </p>

                {hasDonationLinks && (
                  <div className="mt-4 rounded-md bg-black/20 p-3 ring-1 ring-border">
                    <p className="mb-2.5 flex items-center gap-1.5 text-xs text-fg">
                      <Heart className="size-3.5 text-[#db61a2]" /> If FrameScale saves you time, you can chip in.
                    </p>
                    <p className="mb-3 text-[11px] leading-relaxed text-fg-muted">
                      Donations fund GPU test hardware (NVIDIA, AMD, Intel, Apple) and new model support. They never unlock anything.
                    </p>
                    <div className="flex gap-2">
                      {LINKS.sponsors && (
                        <LinkButton href={LINKS.sponsors} icon={Heart} color="#bf3989">
                          GitHub Sponsors
                        </LinkButton>
                      )}
                      {LINKS.kofi && (
                        <LinkButton href={LINKS.kofi} icon={Coffee} color="#ff5e5b">
                          Buy me a coffee on Ko-fi
                        </LinkButton>
                      )}
                    </div>
                  </div>
                )}

                <div className="mt-4 flex items-start gap-2 text-[11px] leading-relaxed text-fg-subtle">
                  <Scale className="mt-0.5 size-3.5 shrink-0" />
                  <p>
                    Licensed under the GNU General Public License v3.0 or later. Bundled components (FFmpeg, ncnn, Real-ESRGAN, RIFE…)
                    keep their own licenses; see THIRD_PARTY_NOTICES.
                    {LINKS.repository && (
                      <>
                        {" "}
                        <button
                          type="button"
                          onClick={() => void backend.openExternal(LINKS.repository)}
                          className="inline-flex items-center gap-1 text-accent hover:text-accent-hover"
                        >
                          <CodeXml className="size-3" /> Source code
                        </button>
                      </>
                    )}
                  </p>
                </div>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        )}
      </AnimatePresence>
    </Dialog.Root>
  );
}
