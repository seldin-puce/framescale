import * as RT from "@radix-ui/react-tooltip";
import type { ReactNode } from "react";

export function TooltipProvider({ children }: { children: ReactNode }) {
  return <RT.Provider delayDuration={350} skipDelayDuration={150}>{children}</RT.Provider>;
}

export function Tooltip({ label, children, side = "bottom" }: { label: ReactNode; children: ReactNode; side?: RT.TooltipContentProps["side"] }) {
  return (
    <RT.Root>
      <RT.Trigger asChild>{children}</RT.Trigger>
      <RT.Portal>
        <RT.Content
          side={side}
          sideOffset={6}
          className="z-50 max-w-64 rounded-sm border border-border-strong bg-surface-raised px-2 py-1 text-xs text-fg shadow-lg shadow-black/40"
        >
          {label}
        </RT.Content>
      </RT.Portal>
    </RT.Root>
  );
}
