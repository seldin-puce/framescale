import { Section } from "../ui/Section";
import { TelemetryHUD } from "../telemetry/TelemetryHUD";
import { OutputSettings } from "./OutputSettings";
import { PipelineChain } from "./PipelineChain";
import { StageInspector } from "./StageInspector";

/** Configuration scrolls; the hardware HUD stays pinned and always visible. */
export function InspectorSidebar() {
  return (
    <aside className="flex h-full flex-col bg-surface-glass" aria-label="Inspector">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <Section title="Pipeline">
          <PipelineChain />
        </Section>
        <Section title="Stage">
          <StageInspector />
        </Section>
        <Section title="Output" className="border-b-0">
          <OutputSettings />
        </Section>
      </div>
      <div className="shrink-0 border-t border-border-strong bg-surface/80 shadow-[0_-8px_16px_-12px_rgba(0,0,0,0.6)]">
        <TelemetryHUD />
      </div>
    </aside>
  );
}
