/**
 * When to show the "Enjoying FrameScale?" card: after the first finished
 * render, then at most once every 10 renders and never more than weekly.
 * Dismissing it pushes the next one back. Local per-user convenience state
 * only. It never gates anything.
 */
import { hasDonationLinks } from "./links";

const KEY = "fs.supportNudge";
const WEEK_MS = 7 * 24 * 3600 * 1000;

interface NudgeState {
  renders: number;
  lastShown: number;
}

function load(): NudgeState {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) ?? "null") as NudgeState | null;
    if (v && typeof v.renders === "number" && typeof v.lastShown === "number") return v;
  } catch {
    // storage unavailable or corrupt: fall through to defaults
  }
  return { renders: 0, lastShown: 0 };
}

function save(s: NudgeState) {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // ignore (private mode, quota)
  }
}

export const supportNudge = {
  /** Record a finished render; true if the card should appear now. */
  due(): boolean {
    if (!hasDonationLinks) return false;
    const s = load();
    s.renders += 1;
    const show = s.renders === 1 || (s.renders % 10 === 0 && Date.now() - s.lastShown > WEEK_MS);
    if (show) s.lastShown = Date.now();
    save(s);
    return show;
  },
  dismissed() {
    const s = load();
    s.lastShown = Date.now();
    save(s);
  },
};
