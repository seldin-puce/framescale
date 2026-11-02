import type { Container, VideoCodec } from "@framescale/ipc-types";
import { useApp } from "@/stores/app";
import { ProGate } from "../pro/ProGate";
import { Tooltip } from "../ui/Tooltip";

const CODECS: { value: VideoCodec; label: string; encoders: string[] }[] = [
  { value: "hevc10", label: "HEVC 10-bit", encoders: ["hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_vaapi", "hevc_videotoolbox", "libkvazaar"] },
  { value: "av1", label: "AV1 10-bit", encoders: ["av1_nvenc", "av1_amf", "av1_qsv", "av1_vaapi", "libsvtav1"] },
  { value: "h264", label: "H.264 (compat)", encoders: ["h264_nvenc", "h264_amf", "h264_qsv", "h264_vaapi", "h264_videotoolbox", "libopenh264"] },
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
              {c.label}{hw(c) ? " · GPU" : available(c) ? " · CPU" : " · unavailable"}
            </option>
          ))}
        </select>
        <select aria-label="Container" className={selectCls} value={output.container} onChange={(e) => setOutput({ container: e.target.value as Container })}>
          <option value="mp4">MP4</option>
          <option value="mkv">MKV</option>
          <option value="mov">MOV</option>
        </select>
      </div>
      <ProGate feature="proResOutput">
        <button type="button" className="w-full rounded-sm border border-border-strong bg-black/20 px-2 py-1.5 text-left text-xs text-fg-muted">
          Apple ProRes 422 HQ / 4444 <span className="text-fg-subtle">(macOS)</span>
        </button>
      </ProGate>
      <label className="flex flex-col gap-1">
        <span className="flex justify-between text-[11px] text-fg-muted">
          <span>Quality</span>
          <span className="tabular font-mono">{output.quality <= 16 ? "Archival" : output.quality <= 22 ? "High" : output.quality <= 28 ? "Balanced" : "Small"} · CQ {output.quality}</span>
        </span>
        <input type="range" min={12} max={34} value={output.quality} onChange={(e) => setOutput({ quality: Number(e.target.value) })} className="accent-accent" />
      </label>
      <Tooltip label="Primaries, transfer (incl. PQ/HLG), matrix and range are carried from source to output. Audio is copied bit-exact.">
        <p className="cursor-help text-[10px] text-fg-subtle">Color &amp; audio passthrough · HDR10 preserved</p>
      </Tooltip>
    </div>
  );
}
