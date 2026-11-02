import type { Feature } from "@framescale/ipc-types";
import type { ReactNode } from "react";
import { cn } from "@/lib/cn";
import { useApp } from "@/stores/app";
import { ProBadge } from "../ui/ProBadge";

/**
 * Pro lock pattern: the control stays visible but dimmed, with a discreet PRO
 * badge, and any interaction opens the activation modal instead. This is a
 * hint only. Rust (host + worker) enforces the same rule.
 */
export function ProGate({ feature, children, className, badge = true }: { feature: Feature; children: ReactNode; className?: string; badge?: boolean }) {
  const unlocked = useApp((s) => s.entitlements.features.includes(feature));
  const openPro = useApp((s) => s.openPro);
  if (unlocked) return <>{children}</>;
  return (
    <div className={cn("relative", className)}>
      <div aria-hidden className="pointer-events-none opacity-45 saturate-50 select-none">
        {children}
      </div>
      <button
        type="button"
        onClick={() => openPro(feature)}
        aria-label={`${feature}: requires FrameScale Pro`}
        className="absolute inset-0 flex items-start justify-end rounded-sm p-1 focus-visible:outline-2 focus-visible:outline-pro"
      >
        {badge && <ProBadge />}
      </button>
    </div>
  );
}
