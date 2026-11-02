import { cn } from "@/lib/cn";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
  className,
}: {
  value: T;
  options: SegmentedOption<T>[];
  onChange: (v: T) => void;
  ariaLabel: string;
  className?: string;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className={cn("flex rounded-sm bg-black/30 p-0.5 ring-1 ring-border", className)}>
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          disabled={o.disabled}
          onClick={() => onChange(o.value)}
          className={cn(
            "flex-1 rounded-[4px] px-2 py-1 text-xs font-medium text-fg-muted transition-colors",
            "hover:text-fg disabled:opacity-40 focus-visible:outline-2 focus-visible:outline-accent",
            value === o.value && "bg-surface-raised text-fg shadow-sm ring-1 ring-border-strong",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
