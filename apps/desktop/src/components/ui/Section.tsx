import type { ReactNode } from "react";
import { cn } from "@/lib/cn";

export function Section({ title, aside, children, className }: { title: string; aside?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={cn("border-b border-border px-3 py-3", className)} aria-label={title}>
      <header className="mb-2 flex items-center justify-between">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.08em] text-fg-subtle">{title}</h2>
        {aside}
      </header>
      {children}
    </section>
  );
}
