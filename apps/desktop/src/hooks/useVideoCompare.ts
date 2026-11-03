import type { CompareInfo, CompareSide } from "@framescale/ipc-types";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ImagePair } from "@/components/viewport/CompareStage";
import { backend } from "@/ipc";
import type { CompareMode } from "@/stores/app";

/** Decoded playback frames kept (both sides); display-sized. */
const CACHE_BYTES = 384 * 1024 * 1024;
/** Frames requested ahead of the playhead (the host decodes ~2 s beyond). */
const PREFETCH = 6;
/** Rest this long on a frame before it's swapped for native-resolution stills. */
const STILL_DELAY_MS = 350;

/** HTTP status from the host: 409 superseded, 416 past the end, 404 closed. */
class FrameError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

const quiet = (e: unknown) => e instanceof FrameError && (e.status === 409 || e.status === 404);

/**
 * Plays a finished render against its source, frame-aligned on the output's
 * frame grid. The host decodes both videos with our own FFmpeg (webviews can't
 * play AV1/HEVC 10-bit/HDR/ProRes reliably) and serves JPEG frames from memory.
 *
 * - Playing: frames are requested a few ahead; playback waits for the next
 *   pair instead of tearing or skipping.
 * - Seeking: one request in flight at a time, latest position wins, so
 *   dragging the timeline never piles up decoders.
 * - Paused: after a moment the frame is replaced by native-resolution stills,
 *   so zooming in shows real pixels, not the playback downscale.
 * - Single-side modes decode only that side.
 */
export function useVideoCompare(info: CompareInfo | null, mode: CompareMode) {
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [pair, setPair] = useState<ImagePair | null>(null);
  const [native, setNative] = useState(false);
  const [buffering, setBuffering] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [restart, setRestart] = useState(0);

  const cache = useRef(new Map<string, Promise<ImageBitmap>>());
  const stills = useRef(new Map<string, Promise<ImageBitmap>>());
  const want = useRef(0); // frame that should be on screen
  const busy = useRef(false); // a seek request is in flight
  const shown = useRef<{ pair: ImagePair; frame: number; full: boolean } | null>(null);
  const playingRef = useRef(false);
  playingRef.current = playing;
  const session = useRef<string | null>(null);
  session.current = info?.session ?? null;

  const sidesKey = mode === "before" ? "before" : mode === "after" ? "after" : "before,after";
  const capacity = info ? Math.max(16, Math.floor(CACHE_BYTES / (info.width * info.height * 4))) : 0;

  const evict = (c: Map<string, Promise<ImageBitmap>>, cap: number) => {
    while (c.size > cap) {
      const oldest = c.keys().next();
      if (oldest.done) break;
      const p = c.get(oldest.value);
      c.delete(oldest.value);
      // Free decoded memory now rather than at the next GC, unless on screen.
      void p?.then(
        (b) => {
          const s = shown.current?.pair;
          if (b !== s?.before && b !== s?.after) b.close();
        },
        () => {},
      );
    }
  };

  const load = useCallback(
    (side: CompareSide, n: number, full: boolean): Promise<ImageBitmap> => {
      if (!info) return Promise.reject(new Error("no comparison"));
      const c = full ? stills.current : cache.current;
      const key = `${side}/${n}`;
      const hit = c.get(key);
      if (hit) {
        c.delete(key); // refresh LRU position
        c.set(key, hit);
        return hit;
      }
      const url = backend.assetUrl(`compare/${info.session}/${side}/${full ? "full/" : ""}${n}.jpg`);
      const p = fetch(url).then(async (r) => {
        if (!r.ok) throw new FrameError(r.status, (await r.text().catch(() => "")) || `HTTP ${r.status}`);
        return createImageBitmap(await r.blob());
      });
      c.set(key, p);
      p.catch(() => {
        if (c.get(key) === p) c.delete(key); // allow a retry
      });
      evict(c, full ? 4 : capacity);
      return p;
    },
    [info, capacity],
  );

  const loadPair = useCallback(
    async (n: number, full: boolean): Promise<ImagePair> => {
      if (!info) throw new Error("no comparison");
      const sides = sidesKey.split(",") as CompareSide[];
      const bitmaps = await Promise.all(sides.map((s) => load(s, n, full)));
      const pick = (s: CompareSide) => bitmaps[sides.indexOf(s)] ?? bitmaps[0]!;
      // Image space: the output's height at its display aspect (anamorphic
      // pixels corrected), so zoom stays put when native stills arrive.
      const width = Math.round((info.outputHeight * info.width) / info.height);
      return { before: pick("before"), after: pick("after"), width, height: info.outputHeight };
    },
    [info, sidesKey, load],
  );

  const show = useCallback((p: ImagePair, n: number, full: boolean) => {
    shown.current = { pair: p, frame: n, full };
    setPair(p);
    setNative(full);
  }, []);

  // A new comparison starts at the first frame with empty caches.
  useEffect(() => {
    for (const c of [cache.current, stills.current]) {
      for (const p of c.values()) void p.then((b) => b.close(), () => {});
      c.clear();
    }
    want.current = 0;
    shown.current = null;
    setFrame(0);
    setPlaying(false);
    setPair(null);
    setNative(false);
    setError(null);
  }, [info]);

  /** Paused seeks: one request in flight, latest position wins. */
  const pump = useCallback(() => {
    if (busy.current || !info || playingRef.current) return;
    const n = want.current;
    const sid = info.session;
    busy.current = true;
    const slow = setTimeout(() => setBuffering(true), 150);
    loadPair(n, false)
      .then((p) => {
        if (session.current !== sid || playingRef.current) return;
        const s = shown.current;
        if (!(s && s.full && s.frame === n)) show(p, n, false); // never downgrade a still
        setError(null);
      })
      .catch((e: unknown) => {
        if (session.current === sid && want.current === n && !quiet(e) && !(e instanceof FrameError && e.status === 416)) {
          setError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        clearTimeout(slow);
        setBuffering(false);
        busy.current = false;
        if (session.current === sid && want.current !== n) pump();
      });
  }, [info, loadPair, show]);

  // Show the current frame whenever paused (first load, mode change, pause).
  useEffect(() => {
    if (info && !playing) pump();
  }, [info, playing, pump]);

  // Playback at the output frame rate; waits for frames rather than skipping.
  useEffect(() => {
    if (!info || !playing) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // Frames are due on an absolute clock, so timer lateness never adds up;
    // after a stall (buffering) the clock restarts instead of rushing ahead.
    const period = 1000 / info.fps;
    let t0 = performance.now();
    let k0 = want.current;
    const step = (k: number) => {
      const next = k + 1;
      if (next >= info.frames) {
        setPlaying(false);
        return;
      }
      for (let i = 2; i <= PREFETCH && k + i < info.frames; i++) void loadPair(k + i, false).catch(() => {});
      const slow = setTimeout(() => live && setBuffering(true), 250);
      loadPair(next, false).then(
        (p) => {
          clearTimeout(slow);
          if (!live) return;
          setBuffering(false);
          const now = performance.now();
          let due = t0 + (next - k0) * period;
          if (now - due > 100) {
            t0 = now;
            k0 = next;
            due = now;
          }
          timer = setTimeout(
            () => {
              if (!live) return;
              want.current = next;
              show(p, next, false);
              setFrame(next);
              step(next);
            },
            Math.max(0, due - performance.now()),
          );
        },
        (e: unknown) => {
          clearTimeout(slow);
          if (!live) return;
          setBuffering(false);
          setPlaying(false);
          if (!quiet(e) && !(e instanceof FrameError && e.status === 416)) setError(e instanceof Error ? e.message : String(e));
        },
      );
    };
    step(want.current);
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, [info, playing, restart, loadPair, show]);

  // Resting on a frame: swap in native-resolution stills for pixel-peeping.
  useEffect(() => {
    if (!info || playing) return;
    if (info.outputWidth <= info.width && info.sourceWidth <= info.width) return; // nothing sharper to get
    let live = true;
    const n = frame;
    const t = setTimeout(() => {
      loadPair(n, true).then(
        (p) => {
          if (live && want.current === n && !playingRef.current) show(p, n, true);
        },
        () => {},
      );
    }, STILL_DELAY_MS);
    return () => {
      live = false;
      clearTimeout(t);
    };
  }, [info, playing, frame, loadPair, show]);

  const seek = useCallback(
    (n: number) => {
      if (!info) return;
      const k = Math.max(0, Math.min(info.frames - 1, Math.round(n)));
      want.current = k;
      setFrame(k);
      if (playingRef.current) setRestart((r) => r + 1);
      else pump();
    },
    [info, pump],
  );

  const togglePlay = useCallback(() => {
    if (!info) return;
    if (!playingRef.current && want.current >= info.frames - 1) {
      want.current = 0; // replay from the start
      setFrame(0);
    }
    setError(null);
    setPlaying((p) => !p);
  }, [info]);

  return { pair, frame, seek, playing, togglePlay, buffering, native, error, frames: info?.frames ?? 0 };
}
