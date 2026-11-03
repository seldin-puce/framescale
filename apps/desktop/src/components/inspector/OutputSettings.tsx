import type { Container, VideoCodec } from "@framescale/ipc-types";
import { FolderOpen, RotateCcw } from "lucide-react";
import { backend } from "@/ipc";
import { prettyDir } from "@/lib/format";
import { CODEC_ENCODERS, containerAccepts, useApp } from "@/stores/app";
import { Tooltip } from "../ui/Tooltip";

const IS_MAC = typeof navigator !== "undefined" && /Mac/.test(navigator.platform);

const CODECS: { value: VideoCodec; label: string; encoders: string[] }[] = [
  { value: "hevc10", label: "HEVC 10-bit", encoders: CODEC_ENCODERS.hevc10 },
  { value: "av1", label: "AV1 10-bit", encoders: CODEC_ENCODERS.av1 },
  { value: "h264", label: "H.264 (compat)", encoders: CODEC_ENCODERS.h264 },
  // ProRes only through Apple's own VideoToolbox encoder (licensed by the OS).
  ...(IS_MAC
    ? [
        { value: "proRes422Hq" as const, label: "ProRes 422 HQ", encoders: ["prores_videotoolbox"] },
        { value: "proRes4444" as const, label: "ProRes 4444", encoders: ["prores_videotoolbox"] },
      ]
    : []),
];

// Stable fallback: a fresh `[]` inside a selector would re-render forever.
const NO_ENCODERS: string[] = [];

const selectCls = "w-full rounded-sm border border-border-strong bg-black/30 px-2 py-1.5 text-xs text-fg outline-none focus:border-accent";

export function OutputSettings() {
  const output = useApp((s) => s.output);
  const setOutput = useApp((s) => s.setOutput);
  const encoders = useApp((s) => s.hardware?.encoders ?? NO_ENCODERS);
  const available = (c: (typeof CODECS)[number]) => encoders.length === 0 || c.encoders.some((e) => encoders.includes(e));
  const hw = (c: (typeof CODECS)[number]) => c.encoders.some((e) => !e.startsWith("lib") && encoders.includes(e));

  return (
    <div className="flex flex-col gap-2.5">
      <div className="grid grid-cols-[1fr_88px] gap-1.5">
        <select aria-label="Video codec" className={selectCls} value={output.codec} onChange={(e) => setOutput({ codec: e.target.value as VideoCodec })}>
          {CODECS.map((c) => (
            <option key={c.value} value={c.value} disabled={!available(c)}>
              {c.label}
              {hw(c)
                ? " · GPU"
                : c.value === "hevc10" && available(c) && !encoders.includes("libx265")
                  ? " · CPU, 8-bit SDR only"
                  : available(c)
                    ? " · CPU"
                    : " · unavailable"}
            </option>
          ))}
        </select>
        <select aria-label="Container" className={selectCls} value={output.container} onChange={(e) => setOutput({ container: e.target.value as Container })}>
          {(["mp4", "mkv", "mov"] as const).map((c) => (
            <option key={c} value={c} disabled={!containerAccepts(c, output.codec)}>
              {c.toUpperCase()}
            </option>
          ))}
        </select>
      </div>
      <label className="flex flex-col gap-1">
        <span className="flex justify-between text-[11px] text-fg-muted">
          <span>Quality</span>
          <span className="tabular font-mono">{output.quality <= 16 ? "Archival" : output.quality <= 22 ? "High" : output.quality <= 28 ? "Balanced" : "Small"} · CQ {output.quality}</span>
        </span>
        <input type="range" min={12} max={34} value={output.quality} onChange={(e) => setOutput({ quality: Number(e.target.value) })} className="accent-accent" />
      </label>
      <SaveTo />
      <Tooltip label="Primaries, transfer (incl. PQ/HLG), matrix and range are carried from source to output. Audio is copied bit-exact.">
        <p className="cursor-help text-[10px] text-fg-subtle">Color &amp; audio passthrough · HDR10 preserved</p>
      </Tooltip>
    </div>
  );
}

/** Where renders go, visible before rendering (not something to hunt for after). */
function SaveTo() {
  const dir = useApp((s) => s.outputDir);
  const setDir = useApp((s) => s.setOutputDir);
  const upscale = useApp((s) => s.stages.upscale);
  const container = useApp((s) => s.output.container);
  const tag = upscale.enabled ? `${upscale.scale}x` : "enhanced";
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] text-fg-muted">Save to</span>
      <div className="flex items-center gap-1.5">
        <div className="min-w-0 flex-1 truncate rounded-sm border border-border-strong bg-black/30 px-2 py-1.5 text-xs text-fg" title={dir ?? undefined}>
          {dir ? prettyDir(`${dir}/x`) : "Same folder as each source"}
        </div>
        <button
          type="button"
          onClick={() => void backend.pickFolder().then((d) => d && setDir(d))}
          className="flex shrink-0 items-center gap-1 rounded-sm bg-white/6 px-2 py-1.5 text-xs text-fg ring-1 ring-border-strong hover:bg-white/10"
        >
          <FolderOpen className="size-3.5" /> Change…
        </button>
        {dir && (
          <button type="button" aria-label="Save next to each source again" title="Save next to each source again" onClick={() => setDir(null)} className="shrink-0 rounded-sm p-1.5 text-fg-subtle hover:bg-white/6 hover:text-fg">
            <RotateCcw className="size-3.5" />
          </button>
        )}
      </div>
      <span className="truncate font-mono text-[10px] text-fg-subtle">
        name_framescale_{tag}.{container}
      </span>
    </div>
  );
}
