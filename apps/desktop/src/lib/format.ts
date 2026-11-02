const GB = 1024 ** 3;
const MB = 1024 ** 2;

export const DASH = "—";

export function bytes(n: number | null | undefined, digits = 1): string {
  if (n == null || !Number.isFinite(n)) return DASH;
  if (n >= GB) return `${(n / GB).toFixed(digits)} GB`;
  if (n >= MB) return `${(n / MB).toFixed(0)} MB`;
  return `${Math.round(n / 1024)} KB`;
}

export function fps(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return DASH;
  return v >= 100 ? v.toFixed(0) : v.toFixed(1);
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds) || seconds < 0) return DASH;
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  const p = (x: number) => x.toString().padStart(2, "0");
  return h > 0 ? `${h}:${p(m)}:${p(ss)}` : `${p(m)}:${p(ss)}`;
}

/** Timecode with centiseconds for the scrubber. */
export function timecode(seconds: number): string {
  const cs = Math.floor((seconds % 1) * 100);
  return `${duration(Math.floor(seconds))}.${cs.toString().padStart(2, "0")}`;
}

export function pct(v: number | null | undefined): string {
  return v == null || !Number.isFinite(v) ? DASH : `${Math.round(v)}%`;
}

export function basename(path: string): string {
  return path.split(/[\\/]/).pop() ?? path;
}

/** `/a/b/clip.mov` → `/a/b/clip_framescale_4x.mkv` */
export function outputPathFor(input: string, scale: number, ext: string): string {
  const sep = input.includes("\\") ? "\\" : "/";
  const dir = input.slice(0, Math.max(0, input.lastIndexOf(sep)));
  const stem = basename(input).replace(/\.[^.]+$/, "");
  const tag = scale > 1 ? `${scale}x` : "enhanced";
  return `${dir}${dir ? sep : ""}${stem}_framescale_${tag}.${ext}`;
}
