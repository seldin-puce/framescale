import { Lock } from "lucide-react";
import { cn } from "@/lib/cn";

export function ProBadge({ locked = true, className }: { locked?: boolean; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 rounded-[4px] px-1 py-px text-[9px] font-bold tracking-wider",
        locked ? "bg-pro/12 text-pro ring-1 ring-pro/30" : "bg-pro text-black",
        className,
      )}
    >
      {locked && <Lock className="size-2.5" strokeWidth={2.5} aria-hidden />}
      PRO
    </span>
  );
}
