import type { PreviewResult } from "@framescale/ipc-types";
import { useCallback, useEffect, useRef, useState } from "react";
import type { ImagePair } from "@/components/viewport/CompareStage";
import { backend } from "@/ipc";

/** Bitmap budget: decoded 4K RGBA is ~33 MB, so the cache size scales with resolution. */
const CACHE_BYTES = 512 * 1024 * 1024;
const PREFETCH = 4;

type Kind = "before" | "after";

/**
 * Plays a rendered preview: decodes JPEG frames off the main thread
 * (`createImageBitmap`), keeps a bounded LRU of bitmaps, prefetches ahead, and
 * only advances when the next before/after pair is ready, so playback never
 * shows a half-loaded pair.
 */
export function usePreviewPlayer(result: PreviewResult | null) {
  const [pair, setPair] = useState<ImagePair | null>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const cache = useRef(new Map<string, Promise<ImageBitmap>>());

  const capacity = result ? Math.max(6, Math.floor(CACHE_BYTES / (result.width * result.height * 4 * 2))) : 0;

  const load = useCallback(
    (kind: Kind, i: number): Promise<ImageBitmap> => {
      if (!result) return Promise.reject(new Error("no preview"));
      const key = `${kind}/${i}`;
      const c = cache.current;
      const hit = c.get(key);
      if (hit) {
        c.delete(key); // refresh LRU position
        c.set(key, hit);
        return hit;
      }
      const url = backend.assetUrl(`${result.prefix}/${kind}/${String(i).padStart(4, "0")}.jpg`);
      const p = fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(`preview frame ${key}: HTTP ${r.status}`);
          return r.blob();
        })
        .then((b) => createImageBitmap(b));
      c.set(key, p);
      while (c.size > capacity * 2) {
        const oldest = c.keys().next().value;
        if (oldest === undefined) break;
        c.delete(oldest);
      }
      return p;
    },
    [result, capacity],
  );

  const beforeIndex = useCallback(
    (k: number) => (result ? Math.min(result.beforeFrames - 1, Math.floor((k * result.beforeFrames) / result.afterFrames)) : 0),
    [result],
  );

  const loadPair = useCallback(
    (k: number) => Promise.all([load("before", beforeIndex(k)), load("after", k)]),
    [load, beforeIndex],
  );

  // New preview → reset to its first frame.
  useEffect(() => {
    cache.current.clear();
    setFrame(0);
    setPlaying(false);
    setPair(null);
  }, [result]);

  // Show the current frame; prefetch the next few.
  useEffect(() => {
    if (!result) return;
    let live = true;
    loadPair(frame)
      .then(([before, after]) => {
        if (live) setPair({ before, after, width: result.width, height: result.height });
      })
      .catch(() => {});
    for (let i = 1; i <= PREFETCH; i++) void loadPair((frame + i) % result.afterFrames).catch(() => {});
    return () => {
      live = false;
    };
  }, [result, frame, loadPair]);

  // Playback at the output frame rate; waits for decode instead of skipping.
  useEffect(() => {
    if (!result || !playing) return;
    let live = true;
    let timer: ReturnType<typeof setTimeout>;
    const step = (k: number) => {
      const next = (k + 1) % result.afterFrames;
      const due = performance.now() + 1000 / result.fps;
      void loadPair(next).then(() => {
        if (!live) return;
        timer = setTimeout(() => {
          setFrame(next);
          step(next);
        }, Math.max(0, due - performance.now()));
      });
    };
    step(frame);
    return () => {
      live = false;
      clearTimeout(timer);
    };
    // `frame` is read once at start; the loop carries its own index.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, playing, loadPair]);

  return { pair, frame, setFrame, playing, setPlaying, frames: result?.afterFrames ?? 0 };
}
