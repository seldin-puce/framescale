import { Columns2, Eye, Maximize2, SquareSplitHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import { useMemo, useState } from "react";
import { makeSamplePair } from "@/gl/sample";
import { useApp, type CompareMode } from "@/stores/app";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";
import { CompareStage } from "./CompareStage";
import { Timeline } from "./Timeline";

const MODES: { value: CompareMode; label: string; icon: typeof Columns2 }[] = [
  { value: "split", label: "Split view", icon: SquareSplitHorizontal },
  { value: "side", label: "Side by side", icon: Columns2 },
  { value: "before", label: "Before only", icon: Eye },
];

export function Viewport() {
  const mode = useApp((s) => s.compareMode);
  const setMode = useApp((s) => s.setCompareMode);
  const selected = useApp((s) => s.selected);
  const draft = useApp((s) => s.drafts.find((d) => d.path === selected));
  const job = useApp((s) => (selected ? s.jobs[selected] : undefined));
  const [zoom, setZoom] = useState(1);
  const [fit, setFit] = useState(0);
  const [zoomReq, setZoomReq] = useState<{ zoom: number; seq: number }>();

  // Until the 2 s preview path lands, show a clearly-labelled sample pair.
  const sample = useMemo(() => makeSamplePair(), []);
  const info = draft?.info ?? null;
  const duration = info?.durationS ?? 60;

  const request = (z: number) => setZoomReq((r) => ({ zoom: z, seq: (r?.seq ?? 0) + 1 }));

  return (
    <div className="flex h-full min-w-0 flex-col gap-2 p-2">
      <div className="flex h-9 items-center gap-1 px-1">
        <div role="toolbar" aria-label="Compare mode" className="flex items-center gap-0.5 rounded-sm bg-black/25 p-0.5 ring-1 ring-border">
          {MODES.map((m) => (
            <IconButton key={m.value} icon={m.icon} label={m.label} active={mode === m.value} onClick={() => setMode(m.value)} size="sm" className="size-7" />
          ))}
        </div>
        <div className="mx-2 h-4 w-px bg-border" />
        <IconButton icon={ZoomOut} label="Zoom out (−)" size="sm" className="size-7" onClick={() => request(zoom / 1.5)} disabled={zoom <= 1.001} />
        <Tooltip label="Zoom: double-click the image or press 0 to fit">
          <span className="tabular w-14 text-center font-mono text-xs text-fg-muted">{zoom <= 1.001 ? "Fit" : `${Math.round(zoom * 100)}%`}</span>
        </Tooltip>
        <IconButton icon={ZoomIn} label="Zoom in (+)" size="sm" className="size-7" onClick={() => request(zoom * 1.5)} />
        <IconButton icon={Maximize2} label="Fit to view (0)" size="sm" className="size-7" onClick={() => setFit((f) => f + 1)} />
        <div className="ml-auto flex min-w-0 items-center gap-2 text-xs text-fg-muted">
          {info && (
            <span className="truncate font-mono">
              {info.video.width}×{info.video.height} · {info.video.codec.toUpperCase()} · {info.video.color.bitDepth}-bit
              {info.video.color.transfer === "pq" ? " · HDR10" : info.video.color.transfer === "hlg" ? " · HLG" : ""}
            </span>
          )}
          <Tooltip label="A procedural sample is shown until a 2 s preview of your clip is rendered.">
            <span className="rounded-[4px] bg-warn/12 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-warn uppercase ring-1 ring-warn/25">
              Sample
            </span>
          </Tooltip>
        </div>
      </div>

      <div className="min-h-0 flex-1">
        <CompareStage
          pair={sample}
          mode={mode}
          beforeLabel="Source"
          afterLabel="FrameScale"
          onZoomChange={setZoom}
          fitSignal={fit}
          {...(zoomReq ? { zoomRequest: zoomReq } : {})}
        />
      </div>

      <Timeline duration={duration} cuts={[]} disabled={!info && !job} />
    </div>
  );
}
