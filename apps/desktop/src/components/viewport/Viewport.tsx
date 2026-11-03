import { Columns2, Eye, Maximize2, SquareSplitHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { makeSamplePair } from "@/gl/sample";
import { usePreviewPlayer } from "@/hooks/usePreviewPlayer";
import { backend } from "@/ipc";
import { timecode } from "@/lib/format";
import { COMING_SOON, STAGE_ORDER, useApp, type CompareMode } from "@/stores/app";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";
import { CompareStage } from "./CompareStage";
import { SNIPPET_S, Timeline } from "./Timeline";

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
  const stages = useApp((s) => s.stages);
  const preview = useApp((s) => s.preview);
  const renderPreview = useApp((s) => s.renderPreview);
  const [zoom, setZoom] = useState(1);
  const [fit, setFit] = useState(0);
  const [zoomReq, setZoomReq] = useState<{ zoom: number; seq: number }>();
  const [t, setT] = useState(0);

  const sample = useMemo(() => makeSamplePair(), []);
  const info = draft?.info ?? null;
  const sourcePath = draft?.path ?? job?.spec.input ?? null;
  const duration = info?.durationS ?? 60;
  useEffect(() => setT(Math.max(0, Math.min(duration - SNIPPET_S, duration * 0.3))), [sourcePath, duration]);

  // A preview is only shown for the clip it was rendered from.
  const current = preview.path === sourcePath ? preview : null;
  const player = usePreviewPlayer(current?.status === "ready" ? current.result : null);
  const r = current?.result ?? null;
  const pair = player.pair ?? sample;
  const showingPreview = player.pair != null;

  const unsupported = STAGE_ORDER.filter((k) => stages[k].enabled && COMING_SOON.has(k));
  const canRender = backend.kind === "tauri" && !!info && unsupported.length === 0;
  const renderHint =
    backend.kind !== "tauri"
      ? "Previews run the real engine: open FrameScale as the desktop app."
      : !info
        ? "Select a clip to preview."
        : unsupported.length
          ? "Turn off stages marked Soon to preview."
          : "Run your exact pipeline on the GPU for 2 s from the marker.";

  const request = (z: number) => setZoomReq((q) => ({ zoom: z, seq: (q?.seq ?? 0) + 1 }));

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
          {showingPreview && r ? (
            <Tooltip
              label={`${r.afterFrames} frames at ${r.width}×${r.height} rendered in ${(r.elapsedMs / 1000).toFixed(1)} s by ${r.backend}${r.device ? ` on ${r.device}` : ""}${r.toneMapped ? ". HDR is tone-mapped for display only; the render keeps HDR." : "."}`}
            >
              <span className="rounded-[4px] bg-accent/12 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-accent uppercase ring-1 ring-accent/25">
                Preview {timecode(r.startS)} · {player.frame + 1}/{player.frames}
              </span>
            </Tooltip>
          ) : (
            <Tooltip label="A procedural sample is shown until a 2 s preview of your clip is rendered.">
              <span className="rounded-[4px] bg-warn/12 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-warn uppercase ring-1 ring-warn/25">
                Sample
              </span>
            </Tooltip>
          )}
        </div>
      </div>

      <div className="relative min-h-0 flex-1">
        <CompareStage
          pair={pair}
          mode={mode}
          beforeLabel="Source"
          afterLabel="FrameScale"
          onZoomChange={setZoom}
          fitSignal={fit}
          {...(zoomReq ? { zoomRequest: zoomReq } : {})}
        />
        {current?.status === "rendering" && (
          <div className="pointer-events-none absolute inset-x-0 top-12 flex justify-center">
            <span className="rounded-sm bg-black/70 px-3 py-1.5 text-xs text-fg backdrop-blur-sm ring-1 ring-border">
              Rendering 2 s on the GPU…
            </span>
          </div>
        )}
      </div>

      <Timeline
        duration={duration}
        t={t}
        onSeek={setT}
        cuts={[]}
        canRender={canRender}
        renderHint={renderHint}
        rendering={current?.status === "rendering"}
        onRender={() => void renderPreview(t)}
        playable={showingPreview}
        playing={player.playing}
        onTogglePlay={() => player.setPlaying((p) => !p)}
        playhead={showingPreview && player.frames ? player.frame / player.frames : null}
      />
    </div>
  );
}
