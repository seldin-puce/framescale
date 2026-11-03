import type { CompareInfo } from "@framescale/ipc-types";
import { Columns2, ExternalLink, Eye, FolderOpen, Maximize2, Sparkles, SquareSplitHorizontal, ZoomIn, ZoomOut } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { makeSamplePair } from "@/gl/sample";
import { usePreviewPlayer } from "@/hooks/usePreviewPlayer";
import { useVideoCompare } from "@/hooks/useVideoCompare";
import { backend, isCmdError } from "@/ipc";
import { duration as fmtDuration, timecode } from "@/lib/format";
import { COMING_SOON, STAGE_ORDER, useApp, type CompareMode, type PreviewRun } from "@/stores/app";
import { IconButton } from "../ui/IconButton";
import { Tooltip } from "../ui/Tooltip";
import { CompareStage } from "./CompareStage";
import { CompareTimeline } from "./CompareTimeline";
import { SNIPPET_S, Timeline, type RenderRun } from "./Timeline";

const fmtFps = (f: number) => `${Number.isInteger(f) ? f : f.toFixed(2)} fps`;

const MODES: { value: CompareMode; label: string; icon: typeof Columns2 }[] = [
  { value: "split", label: "Split view", icon: SquareSplitHorizontal },
  { value: "side", label: "Side by side", icon: Columns2 },
  { value: "before", label: "Original only", icon: Eye },
  { value: "after", label: "Result only", icon: Sparkles },
];

/** Frame size to ask the host for: the viewer's width in device pixels. */
const viewWidth = (el: HTMLElement | null) => Math.min(2560, Math.round((el?.clientWidth ?? 1280) * (window.devicePixelRatio || 1)));

/** "4 s", "1:05". */
function eta(seconds: number): string {
  const s = Math.max(1, Math.ceil(seconds));
  return s < 60 ? `${s} s` : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** What the timeline and the viewer say about a preview render in flight. */
function describeRun(run: PreviewRun): RenderRun & { overlay: string } {
  if (run.phase === "first") {
    return run.started
      ? { fraction: null, label: "First frame…", detail: "Rendering the first frame. It shows as soon as it's ready.", overlay: "Rendering the first frame…" }
      : { fraction: null, label: "Starting…", detail: "Loading the models.", overlay: "Loading the models…" };
  }
  if (!run.started || run.total === 0) {
    return { fraction: 0, label: "0%", detail: "Starting the full 2 s.", overlay: "First frame ready · starting the full 2 s…" };
  }
  const fraction = Math.min(1, run.done / run.total);
  const left = run.fps > 0 && run.done < run.total ? (run.total - run.done) / run.fps : null;
  const pct = `${Math.floor(fraction * 100)}%`;
  return {
    fraction,
    label: left == null ? pct : `${pct} · ${eta(left)} left`,
    detail: `${run.done} of ${run.total} frames${run.fps > 0 ? ` at ${run.fps.toFixed(1)} fps` : ""}.`,
    overlay: `First frame ready · rendering the full 2 s: ${run.done} of ${run.total} frames${left == null ? "" : `, about ${eta(left)} left`}`,
  };
}

export function Viewport() {
  const mode = useApp((s) => s.compareMode);
  const setMode = useApp((s) => s.setCompareMode);
  const selected = useApp((s) => s.selected);
  const draft = useApp((s) => s.drafts.find((d) => d.path === selected));
  const job = useApp((s) => (selected ? s.jobs[selected] : undefined));
  const stages = useApp((s) => s.stages);
  const preview = useApp((s) => s.preview);
  const renderPreview = useApp((s) => s.renderPreview);
  const cancelPreview = useApp((s) => s.cancelPreview);
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
    backend
      .compareOpen(finishedId, viewWidth(stageRef.current))
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

  // Not finished (a new clip, or a job in the queue): show the clip itself,
  // the frame under the marker, decoded by the host like a comparison.
  const viewPath = finished ? null : sourcePath;
  const [src, setSrc] = useState<CompareInfo | null>(null);
  const [srcError, setSrcError] = useState<string | null>(null);
  useEffect(() => {
    setSrc(null);
    setSrcError(null);
    if (!viewPath || backend.kind !== "tauri") return;
    let live = true;
    let session: string | null = null;
    backend
      .sourceOpen(viewPath, viewWidth(stageRef.current))
      .then((i) => {
        session = i.session;
        if (live) setSrc(i);
        else void backend.compareClose(i.session);
      })
      .catch((e: unknown) => live && setSrcError(isCmdError(e) ? e.message : String(e)));
    return () => {
      live = false;
      if (session) void backend.compareClose(session);
    };
  }, [viewPath]);
  const frameAt = (x: number) => (src ? Math.round(x * src.fps) : 0);
  const source = useVideoCompare(src, "before", frameAt(t));
  const seekSource = source.seek;
  useEffect(() => {
    if (src) seekSource(Math.round(t * src.fps));
  }, [src, t, seekSource]);

  const duration = info?.durationS ?? src?.durationS ?? 60;
  useEffect(() => setT(Math.max(0, Math.min(duration - SNIPPET_S, duration * 0.3))), [sourcePath, duration]);

  // A preview is shown for the clip and window it was rendered from; moving
  // the marker elsewhere shows the clip there instead.
  const current = preview.path === sourcePath ? preview : null;
  const r = current?.result ?? null;
  const here = !!r && (src ? frameAt(r.startS) === frameAt(t) : Math.abs(r.startS - t) < 1e-3);
  const player = usePreviewPlayer(current?.status === "ready" && here ? r : null);
  // The window can start earlier than asked near the end: follow it.
  const resultStart = r?.startS ?? null;
  useEffect(() => {
    if (resultStart != null) setT(resultStart);
  }, [r, resultStart]);
  const runView = current?.run ? describeRun(current.run) : null;

  const view: "compare" | "preview" | "source" | "sample" = finished
    ? "compare"
    : here && player.pair
      ? "preview"
      : sourcePath && backend.kind === "tauri"
        ? "source"
        : "sample";
  const pair = view === "compare" ? video.pair : view === "preview" ? player.pair : view === "source" ? source.pair : sample;
  // Nothing to compare yet: the clip fills the viewer.
  const comparable = view !== "source";
  const stageMode: CompareMode = comparable ? mode : "before";

  const unsupported = STAGE_ORDER.filter((k) => stages[k].enabled && COMING_SOON.has(k));
  const canRender = backend.kind === "tauri" && !!info && unsupported.length === 0;
  const renderHint =
    backend.kind !== "tauri"
      ? "Previews run the real engine: open FrameScale as the desktop app."
      : !sourcePath
        ? "Select a clip to preview."
        : !draft
          ? "Previews are for clips that aren't queued yet."
          : !info
            ? draft.error
              ? "FrameScale can't read this clip."
              : "Reading the clip…"
            : unsupported.length
              ? "Turn off stages marked Soon to preview."
              : "Run your exact pipeline on the GPU for 2 s from the marker.";

  const sourceNote = srcError
    ? `Couldn't open the clip: ${srcError}`
    : source.error
      ? `Couldn't decode the clip: ${source.error}`
      : !source.pair
        ? "Opening the clip…"
        : null;
  const overlay = finished ? null : runView ? runView.overlay : view === "source" ? sourceNote : null;
  const overlayError = !runView && !!(srcError || source.error);

  const request = (z: number) => setZoomReq((q) => ({ zoom: z, seq: (q?.seq ?? 0) + 1 }));

  return (
    <div className="flex h-full min-w-0 flex-col gap-2 p-2">
      <div className="flex h-9 items-center gap-1 px-1">
        <div role="toolbar" aria-label="Compare mode" className="flex items-center gap-0.5 rounded-sm bg-black/25 p-0.5 ring-1 ring-border">
          {MODES.map((m) => (
            <IconButton
              key={m.value}
              icon={m.icon}
              label={comparable ? m.label : `${m.label} (after a preview)`}
              active={comparable && mode === m.value}
              disabled={!comparable}
              onClick={() => setMode(m.value)}
              size="sm"
              className="size-7"
            />
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
          {info ? (
            <span className="truncate font-mono">
              {info.video.width}×{info.video.height} · {info.video.codec.toUpperCase()} · {info.video.color.bitDepth}-bit
              {info.video.color.transfer === "pq" ? " · HDR10" : info.video.color.transfer === "hlg" ? " · HLG" : ""}
            </span>
          ) : (
            view === "source" &&
            src && (
              <span className="truncate font-mono">
                {src.sourceWidth}×{src.sourceHeight} · {fmtFps(src.sourceFps)}
              </span>
            )
          )}
          {view === "compare" && finished ? (
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
          ) : view === "preview" && r ? (
            <Tooltip
              label={`${r.afterFrames} frames at ${r.width}×${r.height} rendered in ${(r.elapsedMs / 1000).toFixed(1)} s by ${r.backend}${r.device ? ` on ${r.device}` : ""}${r.toneMapped ? ". HDR is tone-mapped for display only; the render keeps HDR." : "."}`}
            >
              <span className="rounded-[4px] bg-accent/12 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider whitespace-nowrap text-accent uppercase ring-1 ring-accent/25">
                Preview {timecode(r.startS)} · {player.frame + 1}/{player.frames}
              </span>
            </Tooltip>
          ) : view === "source" ? (
            <Tooltip
              label={
                draft
                  ? `Your clip at the marker. Drag the marker to look around, then Render 2 s to see the result here.${src?.toneMapped ? " HDR is tone-mapped for display." : ""}${source.native ? " Full resolution." : ""}`
                  : "The original clip. The result shows here when its render finishes."
              }
            >
              <span className="rounded-[4px] bg-white/6 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider whitespace-nowrap text-fg-muted uppercase ring-1 ring-border">
                Source{source.native ? " · Full res" : ""}
              </span>
            </Tooltip>
          ) : (
            <Tooltip
              label={
                backend.kind === "tauri"
                  ? "A procedural sample is shown until you add a clip."
                  : "The browser demo shows a procedural sample; your clips show in the desktop app."
              }
            >
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
          mode={stageMode}
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
        {overlay && (
          <div className="pointer-events-none absolute inset-x-0 top-12 flex justify-center px-4" aria-live="polite">
            <span className={`max-w-xl rounded-sm bg-black/75 px-3 py-1.5 text-xs backdrop-blur-sm ring-1 ${overlayError ? "text-err ring-err/40" : "text-fg ring-border"}`}>
              {overlay}
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
          scrubbable={!!src || canRender}
          canRender={canRender}
          renderHint={renderHint}
          run={runView}
          onRender={() => void renderPreview(t)}
          onCancel={cancelPreview}
          playable={view === "preview" && (r?.afterFrames ?? 0) > 1}
          playing={player.playing}
          onTogglePlay={() => player.setPlaying((p) => !p)}
          playhead={view === "preview" && player.frames ? player.frame / player.frames : null}
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
