import { GripVertical } from "lucide-react";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { CompareRenderer, type Mode, type View } from "@/gl/compare";
import { cn } from "@/lib/cn";
import type { CompareMode } from "@/stores/app";

export interface ImagePair {
  before: TexImageSource;
  after: TexImageSource;
  width: number;
  height: number;
}

const MODES: Record<CompareMode, Mode> = { split: 0, side: 1, before: 2, after: 3 };
const MIN_ZOOM = 1;
const MAX_ZOOM = 32;

interface Props {
  pair: ImagePair | null;
  mode: CompareMode;
  beforeLabel?: string;
  afterLabel?: string;
  /** Reports zoom (1 = fit) so the toolbar can show it. */
  onZoomChange?: (zoom: number) => void;
  /** Bump to reset zoom/pan to fit. */
  fitSignal?: number;
  /** Bump with a target zoom to set it from the toolbar. */
  zoomRequest?: { zoom: number; seq: number };
  className?: string;
}

/**
 * Interactive before/after viewer.
 *
 * Interaction:
 * - drag the handle, or anywhere in split mode at 1×, to move the split
 * - wheel / pinch: zoom around the cursor; drag (or Space+drag) to pan when zoomed
 * - keyboard on the handle: ←/→ (Shift = 10 %), Home/End; +/−/0 for zoom
 */
export function CompareStage({ pair, mode, beforeLabel = "Before", afterLabel = "After", onZoomChange, fitSignal, zoomRequest, className }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<CompareRenderer | null>(null);
  const view = useRef<View>({ mode: MODES[mode], split: 0.5, zoom: 1, panX: 0, panY: 0 });
  const frame = useRef(0);
  const [split, setSplit] = useState(0.5); // mirrored for the DOM handle only
  const [error, setError] = useState<string | null>(null);
  const [panning, setPanning] = useState(false);
  const spaceDown = useRef(false);

  const draw = useCallback(() => {
    cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => rendererRef.current?.draw(view.current));
  }, []);

  // Renderer lifecycle.
  useEffect(() => {
    const c = canvasRef.current;
    if (!c) return;
    try {
      rendererRef.current = new CompareRenderer(c);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
    return () => {
      cancelAnimationFrame(frame.current);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!pair || !rendererRef.current) return;
    rendererRef.current.setImages(pair.before, pair.after, pair.width, pair.height);
    draw();
  }, [pair, draw]);

  useEffect(() => {
    view.current.mode = MODES[mode];
    draw();
  }, [mode, draw]);

  // Device-pixel-exact canvas sizing.
  useLayoutEffect(() => {
    const wrap = wrapRef.current;
    const c = canvasRef.current;
    if (!wrap || !c) return;
    const ro = new ResizeObserver(([entry]) => {
      if (!entry) return;
      const dpr = window.devicePixelRatio || 1;
      const box = entry.devicePixelContentBoxSize?.[0];
      c.width = Math.max(1, Math.round(box ? box.inlineSize : entry.contentRect.width * dpr));
      c.height = Math.max(1, Math.round(box ? box.blockSize : entry.contentRect.height * dpr));
      draw();
    });
    try {
      ro.observe(wrap, { box: "device-pixel-content-box" });
    } catch {
      ro.observe(wrap); // WebKit without device-pixel-content-box
    }
    return () => ro.disconnect();
  }, [draw]);

  const clampPan = () => {
    const r = rendererRef.current;
    if (!r) return;
    const v = view.current;
    const [iw, ih] = r.image;
    // Allow panning until the image edge reaches the view centre.
    v.panX = Math.max(-iw / 2, Math.min(iw / 2, v.panX));
    v.panY = Math.max(-ih / 2, Math.min(ih / 2, v.panY));
    if (v.zoom <= MIN_ZOOM + 1e-3) {
      v.panX = 0;
      v.panY = 0;
    }
  };

  const setZoom = useCallback(
    (z: number, anchor?: { x: number; y: number }) => {
      const r = rendererRef.current;
      const c = canvasRef.current;
      if (!r || !c) return;
      const v = view.current;
      const next = Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
      // Keep the image point under the anchor fixed: pan += d · (1/s − 1/s').
      const halfW = v.mode === 1 ? c.width / 2 : c.width;
      const ax = anchor ? (v.mode === 1 ? anchor.x % halfW : anchor.x) - halfW / 2 : 0;
      const ay = anchor ? anchor.y - c.height / 2 : 0;
      const s0 = r.scaleFor(v);
      v.zoom = next;
      const s1 = r.scaleFor(v);
      v.panX += ax * (1 / s0 - 1 / s1);
      v.panY += ay * (1 / s0 - 1 / s1);
      clampPan();
      onZoomChange?.(v.zoom);
      draw();
    },
    [draw, onZoomChange],
  );

  useEffect(() => {
    if (fitSignal === undefined) return;
    view.current.zoom = 1;
    view.current.panX = 0;
    view.current.panY = 0;
    onZoomChange?.(1);
    draw();
  }, [fitSignal, draw, onZoomChange]);

  useEffect(() => {
    if (zoomRequest) setZoom(zoomRequest.zoom);
  }, [zoomRequest, setZoom]);

  const toCanvas = (e: { clientX: number; clientY: number }) => {
    const c = canvasRef.current;
    if (!c) return { x: 0, y: 0, fx: 0 };
    const rect = c.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    return { x: fx * c.width, y: ((e.clientY - rect.top) / rect.height) * c.height, fx };
  };

  const updateSplit = (fx: number) => {
    const s = Math.max(0, Math.min(1, fx));
    view.current.split = s;
    setSplit(s);
    draw();
  };

  // Native, non-passive wheel listener (React's is passive and can't preventDefault).
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      setZoom(view.current.zoom * factor, toCanvas(e));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [setZoom]);

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.code === "Space" && !(e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement)) spaceDown.current = true;
    };
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") spaceDown.current = false;
    };
    window.addEventListener("keydown", down);
    window.addEventListener("keyup", up);
    return () => {
      window.removeEventListener("keydown", down);
      window.removeEventListener("keyup", up);
    };
  }, []);

  const drag = useRef<{ kind: "split" | "pan"; x: number; y: number } | null>(null);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 1) return;
    const p = toCanvas(e);
    const v = view.current;
    const wantsPan = e.button === 1 || spaceDown.current || (v.zoom > MIN_ZOOM + 1e-3 && !(e.target as HTMLElement).dataset.handle);
    const kind = wantsPan ? "pan" : v.mode === 0 ? "split" : null;
    if (!kind) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { kind, x: e.clientX, y: e.clientY };
    if (kind === "pan") setPanning(true);
    else updateSplit(p.fx);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    if (d.kind === "split") {
      updateSplit(toCanvas(e).fx);
    } else {
      const r = rendererRef.current;
      if (!r) return;
      const dpr = window.devicePixelRatio || 1;
      const s = r.scaleFor(view.current);
      view.current.panX -= ((e.clientX - d.x) * dpr) / s;
      view.current.panY -= ((e.clientY - d.y) * dpr) / s;
      d.x = e.clientX;
      d.y = e.clientY;
      clampPan();
      draw();
    }
  };

  const endDrag = () => {
    drag.current = null;
    setPanning(false);
  };

  const onHandleKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 0.1 : 0.01;
    const keys: Record<string, () => void> = {
      ArrowLeft: () => updateSplit(view.current.split - step),
      ArrowRight: () => updateSplit(view.current.split + step),
      Home: () => updateSplit(0),
      End: () => updateSplit(1),
      "+": () => setZoom(view.current.zoom * 1.25),
      "=": () => setZoom(view.current.zoom * 1.25),
      "-": () => setZoom(view.current.zoom / 1.25),
      "0": () => setZoom(1),
    };
    const fn = keys[e.key];
    if (fn) {
      e.preventDefault();
      fn();
    }
  };

  const zoomed = view.current.zoom > MIN_ZOOM + 1e-3;

  return (
    <div
      ref={wrapRef}
      className={cn(
        "relative h-full w-full overflow-hidden rounded-md bg-bg ring-1 ring-border select-none touch-none",
        panning ? "cursor-grabbing" : zoomed ? "cursor-grab" : mode === "split" ? "cursor-ew-resize" : "cursor-default",
        className,
      )}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
      onDoubleClick={() => setZoom(1)}
    >
      <canvas ref={canvasRef} className="absolute inset-0 h-full w-full" aria-hidden />

      {error && (
        <div className="absolute inset-0 grid place-items-center text-sm text-err">Viewer unavailable: {error}</div>
      )}

      {/* Corner labels */}
      {pair && mode !== "after" && (
        <span className="pointer-events-none absolute top-2.5 left-2.5 rounded-[4px] bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-fg-muted uppercase backdrop-blur-sm">
          {beforeLabel}
        </span>
      )}
      {pair && mode !== "before" && (
        <span className="pointer-events-none absolute top-2.5 right-2.5 rounded-[4px] bg-black/55 px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-accent uppercase backdrop-blur-sm">
          {afterLabel}
        </span>
      )}
      {mode === "side" && <div className="pointer-events-none absolute inset-y-0 left-1/2 w-px bg-border-strong" />}

      {/* Split handle (DOM, for accessibility and crisp rendering) */}
      {pair && mode === "split" && (
        <div className="pointer-events-none absolute inset-y-0" style={{ left: `${split * 100}%` }}>
          <div className="absolute inset-y-0 -translate-x-1/2 w-px bg-white/85 shadow-[0_0_0_1px_rgba(0,0,0,0.35)]" />
          <div
            role="slider"
            tabIndex={0}
            data-handle="1"
            aria-label="Before/after split"
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(split * 100)}
            aria-valuetext={`${Math.round(split * 100)}% before`}
            onKeyDown={onHandleKey}
            onPointerDown={(e) => {
              e.stopPropagation();
              (e.currentTarget.parentElement?.parentElement as HTMLElement | null)?.setPointerCapture(e.pointerId);
              drag.current = { kind: "split", x: e.clientX, y: e.clientY };
            }}
            className="pointer-events-auto absolute top-1/2 grid h-10 w-6 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize place-items-center rounded-full bg-white text-black shadow-lg shadow-black/50 ring-1 ring-black/20 transition-transform hover:scale-105 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent"
          >
            <GripVertical className="size-3.5" strokeWidth={2.25} aria-hidden />
          </div>
        </div>
      )}
    </div>
  );
}
