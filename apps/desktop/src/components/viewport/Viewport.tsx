import type { CompareInfo } from "@framescale/ipc-types";
import { Columns2, ExternalLink, Eye, FolderOpen, Maximize2, Sparkles, SquareSplitHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeSamplePair } from "@/gl/sample";
import { usePreviewPlayer } from "@/hooks/usePreviewPlayer";
import { useVideoCompare } from "@/hooks/useVideoCompare";
import { backend, isCmdError } from "@/ipc";
import { duration as fmtDuration, timecode } from "@/lib/format";
import { COMING_SOON, STAGE_ORDER, useApp, type CompareMode } from "@/stores/app";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";
import { CompareStage } from "./CompareStage";
import { CompareTimeline } from "./CompareTimeline";
import { SNIPPET_S, Timeline } from "./Timeline";

const fmtFps = (f: number) => `${Number.isInteger(f) ? f : f.toFixed(2)} fps`;

const MODES: { value: CompareMode; label: string; icon: typeof Columns2 }[] = [
  { value: "split", label: "Split view", icon: SquareSplitHorizontal },
  { value: "side", label: "Side by side", icon: Columns2 },
  { value: "before", label: "Original only", icon: Eye },
  { value: "after", label: "Result only", icon: Sparkles },
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
  const stageRef = useRef<HTMLDivElement>(null);

  // A finished job: compare the whole render with its source.
  const finished = job?.state === "done" ? job : undefined;
  const [cmp, setCmp] = useState<CompareInfo | null>(null);
  const [cmpError, setCmpError] = useState<string | null>(null);
  const finishedId = finished?.id;
  useEffect(() => {
    setCmp(null);
    setCmpError(null);
    if (!finishedId || backend.kind !== "tauri") return;
    let live = true;
    let session: string | null = null;
    const px = (stageRef.current?.clientWidth ?? 1280) * (window.devicePixelRatio || 1);
    backend
      .compareOpen(finishedId, Math.min(2560, Math.round(px)))
      .then((i) => {
        session = i.session;
        if (live) setCmp(i);
        else void backend.compareClose(i.session);
      })
      .catch((e: unknown) => live && setCmpError(isCmdError(e) ? e.message : String(e)));
    return () => {
      live = false;
      if (session) void backend.compareClose(session); // stop its decoders
    };
  }, [finishedId]);
  const video = useVideoCompare(cmp, mode);

  // "Play & compare" (completion card, queue): start once the comparison is open.
  const autoplay = useApp((s) => s.autoplay);
  useEffect(() => {
    if (!cmp || !finishedId || autoplay !== finishedId) return;
    useApp.getState().clearAutoplay();
    if (!keys.current.video.playing) keys.current.video.togglePlay();
  }, [cmp, finishedId, autoplay]);

  // Transport keys while comparing: Space play/pause, ←/→ or ,/. frame step (Shift: 1 s).
  const keys = useRef({ video, fps: cmp?.fps ?? 24 });
  keys.current = { video, fps: cmp?.fps ?? 24 };
  useEffect(() => {
    if (!cmp) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      const el = e.target instanceof HTMLElement ? e.target : null;
      if (el?.closest("input, textarea, select, [contenteditable='true'], [role='dialog']")) return;
      const { video: v, fps } = keys.current;
      const step = e.shiftKey ? Math.round(fps) : 1;
      if (e.key === " " && !el?.closest("button")) v.togglePlay();
      else if (e.key === "ArrowLeft" || e.key === ",") v.seek(v.frame - step);
      else if (e.key === "ArrowRight" || e.key === ".") v.seek(v.frame + step);
      else return;
      e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cmp]);

  const sample = useMemo(() => makeSamplePair(), []);
  const info = draft?.info ?? null;
  const sourcePath = draft?.path ?? job?.spec.input ?? null;
  const duration = info?.durationS ?? 60;
  useEffect(() => setT(Math.max(0, Math.min(duration - SNIPPET_S, duration * 0.3))), [sourcePath, duration]);

  // A preview is only shown for the clip it was rendered from.
  const current = preview.path === sourcePath ? preview : null;
  const player = usePreviewPlayer(current?.status === "ready" ? current.result : null);
  const r = current?.result ?? null;
  const pair = (finished ? video.pair : player.pair) ?? sample;
  const showingPreview = !finished && player.pair != null;

  const unsupported = STAGE_ORDER.filter((k) => stages[k].enabled && COMING_SOON.has(k));
  const needsFaceModel = stages.faceFix.enabled && !stages.faceFix.modelPath;
  const canRender = backend.kind === "tauri" && !!info && unsupported.length === 0 && !needsFaceModel;
  const renderHint =
    backend.kind !== "tauri"
      ? "Previews run the real engine: open FrameScale as the desktop app."
      : !info
        ? "Select a clip to preview."
        : unsupported.length
          ? "Turn off stages marked Soon to preview."
          : needsFaceModel
            ? "Face Fix needs your CodeFormer model folder (see the Face Fix stage)."
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
          {finished ? (
            <>
              {cmp && (
                <span className="truncate font-mono">
                  {cmp.sourceWidth}×{cmp.sourceHeight} · {fmtFps(cmp.sourceFps)} → {cmp.outputWidth}×{cmp.outputHeight} · {fmtFps(cmp.fps)}
                </span>
              )}
              <Tooltip
                label={`Both videos play frame-aligned at the result's frame rate.${video.native ? " Paused: full resolution." : " Pause to see full resolution."}${cmp?.toneMapped ? " HDR is tone-mapped for display; the files keep HDR." : ""}`}
              >
                <span className="rounded-[4px] bg-ok/12 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider whitespace-nowrap text-ok uppercase ring-1 ring-ok/25">
                  Original vs result{cmp ? ` · ${fmtDuration(cmp.durationS)}` : ""}
                  {video.native ? " · Full res" : ""}
                </span>
              </Tooltip>
              <div className="flex items-center gap-0.5">
                <ExternalButton label="Original" title="Open the original in your video player" onClick={() => void useApp.getState().openSource(finished.id)} />
                <ExternalButton label="Result" title="Open the result in your video player" onClick={() => void useApp.getState().openOutput(finished.id)} />
                <IconButton icon={FolderOpen} label="Show the result in its folder" size="sm" className="size-7" onClick={() => void useApp.getState().revealOutput(finished.id)} />
              </div>
            </>
          ) : showingPreview && r ? (
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

      <div ref={stageRef} className="relative min-h-0 flex-1">
        <CompareStage
          pair={pair}
          mode={mode}
          beforeLabel={finished ? "Original" : "Source"}
          afterLabel="FrameScale"
          onZoomChange={setZoom}
          fitSignal={fit}
          {...(zoomReq ? { zoomRequest: zoomReq } : {})}
        />
        {finished && (cmpError || video.error || !video.pair) && (
          <div className="pointer-events-none absolute inset-x-0 top-12 flex justify-center px-4">
            <span className={`max-w-xl rounded-sm bg-black/75 px-3 py-1.5 text-xs backdrop-blur-sm ring-1 ${cmpError || video.error ? "text-err ring-err/40" : "text-fg ring-border"}`}>
              {cmpError ??
                (video.error
                  ? `Couldn't decode the videos: ${video.error}`
                  : backend.kind === "tauri"
                    ? "Loading the comparison…"
                    : "Comparing finished renders needs the desktop app.")}
            </span>
          </div>
        )}
        {!finished && (current?.status === "rendering" || current?.refining) && (
          <div className="pointer-events-none absolute inset-x-0 top-12 flex justify-center">
            <span className="rounded-sm bg-black/70 px-3 py-1.5 text-xs text-fg backdrop-blur-sm ring-1 ring-border">
              {current.refining ? "First frame ready: rendering the full 2 s on the GPU…" : "Rendering the first frame…"}
            </span>
          </div>
        )}
      </div>

      {finished ? (
        <CompareTimeline
          frames={video.frames}
          fps={cmp?.fps ?? 24}
          frame={video.frame}
          onSeek={video.seek}
          playing={video.playing}
          onTogglePlay={video.togglePlay}
          buffering={video.buffering}
        />
      ) : (
      <Timeline
        duration={duration}
        t={t}
        onSeek={setT}
        cuts={[]}
        canRender={canRender}
        renderHint={renderHint}
        rendering={current?.status === "rendering" || !!current?.refining}
        onRender={() => void renderPreview(t)}
        playable={showingPreview && (r?.afterFrames ?? 0) > 1}
        playing={player.playing}
        onTogglePlay={() => player.setPlaying((p) => !p)}
        playhead={showingPreview && player.frames ? player.frame / player.frames : null}
      />
      )}
    </div>
  );
}

/** Opens a file in the system's video player (outside the app). */
function ExternalButton({ label, title, onClick }: { label: string; title: string; onClick: () => void }) {
  return (
    <Tooltip label={title}>
      <button
        type="button"
        aria-label={title}
        onClick={onClick}
        className="flex h-7 items-center gap-1 rounded-sm px-1.5 text-[11px] text-fg-muted hover:bg-white/6 hover:text-fg focus-visible:outline-2 focus-visible:outline-accent"
      >
        <ExternalLink className="size-3" /> {label}
      </button>
    </Tooltip>
  );
}
