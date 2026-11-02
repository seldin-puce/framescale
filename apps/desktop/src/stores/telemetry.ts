/**
 * Telemetry lives OUTSIDE React state: samples go into a ring buffer, and
 * components read a snapshot that refreshes at most 4×/s. A 2 Hz stream
 * therefore causes ≤4 renders/s app-wide, and the sparkline draws straight
 * to canvas from the ring.
 */
import type { Telemetry } from "@framescale/ipc-types";
import { useSyncExternalStore } from "react";

const CAPACITY = 120; // 60 s at 2 Hz
const MIN_INTERVAL_MS = 250;

const ring: (Telemetry | undefined)[] = new Array(CAPACITY);
let head = 0;
let count = 0;
let snapshot: Telemetry | null = null;
let lastPublish = 0;
let pending: ReturnType<typeof setTimeout> | null = null;
const listeners = new Set<() => void>();

function publish() {
  pending = null;
  lastPublish = performance.now();
  snapshot = latest();
  listeners.forEach((l) => l());
}

export function pushTelemetry(t: Telemetry): void {
  ring[head] = t;
  head = (head + 1) % CAPACITY;
  count = Math.min(count + 1, CAPACITY);
  const wait = MIN_INTERVAL_MS - (performance.now() - lastPublish);
  if (wait <= 0) publish();
  else if (!pending) pending = setTimeout(publish, wait);
}

export function latest(): Telemetry | null {
  return count === 0 ? null : (ring[(head - 1 + CAPACITY) % CAPACITY] ?? null);
}

/** Oldest → newest values of one numeric field (for sparklines). */
export function series(pick: (t: Telemetry) => number | null | undefined): (number | null)[] {
  const out: (number | null)[] = [];
  for (let i = 0; i < count; i++) {
    const t = ring[(head - count + i + CAPACITY) % CAPACITY];
    out.push(t ? (pick(t) ?? null) : null);
  }
  return out;
}

export const CAPACITY_SAMPLES = CAPACITY;

function subscribe(cb: () => void) {
  listeners.add(cb);
  return () => listeners.delete(cb);
}

export function useTelemetry(): Telemetry | null {
  return useSyncExternalStore(subscribe, () => snapshot, () => null);
}
