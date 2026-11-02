import type { LucideIcon } from "lucide-react";
import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/cn";
import { Tooltip } from "./Tooltip";

interface Props extends ButtonHTMLAttributes<HTMLButtonElement> {
  icon: LucideIcon;
  label: string;
  active?: boolean;
  size?: "sm" | "md";
}

/** Icon-only button: the label becomes the tooltip AND the accessible name. */
export const IconButton = forwardRef<HTMLButtonElement, Props>(function IconButton(
  { icon: Icon, label, active, size = "md", className, ...rest },
  ref,
) {
  return (
    <Tooltip label={label}>
      <button
        ref={ref}
        type="button"
        aria-label={label}
        aria-pressed={active}
        className={cn(
          "inline-flex shrink-0 items-center justify-center rounded-sm text-fg-muted transition-colors",
          "hover:bg-white/6 hover:text-fg focus-visible:outline-2 focus-visible:outline-accent disabled:pointer-events-none disabled:opacity-40",
          size === "md" ? "size-8" : "size-6",
          active && "bg-accent/15 text-accent hover:bg-accent/20 hover:text-accent",
          className,
        )}
        {...rest}
      >
        <Icon className={size === "md" ? "size-4" : "size-3.5"} strokeWidth={1.75} />
      </button>
    </Tooltip>
  );
});
