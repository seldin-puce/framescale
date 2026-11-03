import { Group, Panel, Separator } from "react-resizable-panels";
import { InspectorSidebar } from "../inspector/InspectorSidebar";
import { QueueSidebar } from "../queue/QueueSidebar";
import { Viewport } from "../viewport/Viewport";
import { AboutDialog } from "../support/AboutDialog";
import { CompletionCard } from "../support/CompletionCard";
import { StatusBar } from "./StatusBar";
import { TitleBar } from "./TitleBar";
import { Toast } from "./Toast";

function Handle() {
  return (
    <Separator className="group relative w-px bg-border outline-none data-[separator=active]:bg-accent data-[separator=hover]:bg-border-strong">
      <span className="absolute inset-y-0 -left-1 -right-1" />
    </Separator>
  );
}

/** Three-column workspace: Queue · Viewport · Inspector. */
export function AppShell() {
  return (
    <div className="flex h-full flex-col">
      <TitleBar />
      <Group orientation="horizontal" className="min-h-0 flex-1">
        <Panel id="queue" defaultSize="280px" minSize="220px" maxSize="400px" collapsible collapsedSize="0px">
          <QueueSidebar />
        </Panel>
        <Handle />
        <Panel id="viewport" minSize="480px">
          <main className="h-full" aria-label="Preview">
            <Viewport />
          </main>
        </Panel>
        <Handle />
        <Panel id="inspector" defaultSize="340px" minSize="300px" maxSize="440px">
          <InspectorSidebar />
        </Panel>
      </Group>
      <StatusBar />
      <AboutDialog />
      <CompletionCard />
      <Toast />
    </div>
  );
}
