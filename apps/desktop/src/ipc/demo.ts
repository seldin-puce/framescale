/**
 * Browser-only simulation of the Rust host, so the UI can be developed and
 * reviewed without Tauri. Clearly labelled "Demo data" in the title bar.
 */
import type {
  DeviceUuid,
  EcoMode,
  GpuDevice,
  JobEvent,
  JobId,
  JobState,
  JobSummary,
  MediaInfo,
  Telemetry,
} from "@framescale/ipc-types";
import type { Backend, CmdError } from "./backend";

const GiB = 1024 ** 3;

const devices: GpuDevice[] = [
  {
    uuid: "4e564944494147454654523430373000",
    name: "NVIDIA GeForce RTX 4070",
    vendor: "nvidia",
    pciVendorId: 0x10de,
    pciDeviceId: 0x2786,
    kind: "discrete",
    driverName: "NVIDIA",
    driverInfo: "566.36",
    apiVersion: "1.3.289",
    vramBytes: 12 * GiB,
    unifiedMemory: false,
    fp16: true,
    cooperativeMatrix: true,
    subgroupMin: 32,
    subgroupMax: 32,
    memoryBudget: true,
    drmRenderMinor: null,
    score: 5200,
  },
  {
    uuid: "696e74656c2d697269732d78652d3936",
    name: "Intel Iris Xe Graphics",
    vendor: "intel",
    pciVendorId: 0x8086,
    pciDeviceId: 0x9a49,
    kind: "integrated",
    driverName: "Intel open-source Mesa driver",
    driverInfo: "Mesa 25.2",
    apiVersion: "1.4.305",
    vramBytes: 16 * GiB,
    unifiedMemory: true,
    fp16: true,
    cooperativeMatrix: false,
    subgroupMin: 8,
    subgroupMax: 32,
    memoryBudget: true,
    drmRenderMinor: 128,
    score: 1600,
  },
];


function fakeInfo(path: string, i: number): MediaInfo {
  const hdr = i % 3 === 1;
  return {
    path,
    format: "mov,mp4,m4a,3gp,3g2,mj2",
    durationS: 42 + i * 17.5,
    sizeBytes: 380_000_000 + i * 90_000_000,
    bitRate: 24_000_000,
    startTime: 0,
    video: {
      index: 0,
      codec: hdr ? "hevc" : "h264",
      profile: hdr ? "Main 10" : "High",
      width: hdr ? 3840 : 1920,
      height: hdr ? 2160 : 1080,
      sampleAspect: { num: 1, den: 1 },
      rFrameRate: { num: 24000, den: 1001 },
      avgFrameRate: { num: 24000, den: 1001 },
      frameCount: null,
      startTime: 0,
      pixFmt: hdr ? "yuv420p10le" : "yuv420p",
      color: {
        primaries: hdr ? "bt2020" : "bt709",
        transfer: hdr ? "pq" : "bt709",
        matrix: hdr ? "bt2020Nc" : "bt709",
        range: "limited",
        bitDepth: hdr ? 10 : 8,
        mastering: null,
        contentLight: null,
        inferred: false,
      },
      rotation: 0,
      fieldOrder: "progressive",
      dolbyVision: false,
      hdr10Plus: false,
    },
    audio: [{ index: 1, codec: "aac", channels: 2, sampleRate: 48000, startTime: 0, language: "eng" }],
    subtitles: [],
    chapters: 0,
  };
}

type Sim = { summary: JobSummary; paused: boolean };

export function createDemoBackend(): Backend {
  let selected: DeviceUuid | null = null;
  let eco: EcoMode = { mode: "off" };
  const jobs = new Map<JobId, Sim>();
  const jobSubs = new Set<(e: JobEvent) => void>();
  const emit = (e: JobEvent) => jobSubs.forEach((cb) => cb(e));
  let renderFps: number | null = null;

  const setState = (s: Sim, state: JobState, reason: string | null = null) => {
    s.summary.state = state;
    emit({ type: "stateChanged", job: s.summary.id, state, reason });
  };

  // Job simulator: one job at a time, like the real queue.
  setInterval(() => {
    const all = [...jobs.values()].sort((a, b) => a.summary.queuePos - b.summary.queuePos);
    let active = all.find((s) => ["analyzing", "rendering", "paused", "muxing"].includes(s.summary.state));
    if (!active) {
      const next = all.find((s) => s.summary.state === "queued");
      if (next) setState(next, "analyzing");
      return;
    }
    const s = active.summary;
    if (s.state === "analyzing") {
      s.framesTotal = 2400;
      setState(active, "rendering");
    } else if (s.state === "rendering" && !active.paused) {
      const fpsNow = 11 + Math.random() * 3;
      renderFps = fpsNow;
      s.framesDone = Math.min(s.framesTotal ?? 0, s.framesDone + Math.round(fpsNow * 0.5 * 8));
      const total = s.framesTotal ?? 1;
      emit({
        type: "progress",
        job: s.id,
        framesDone: s.framesDone,
        framesTotal: total,
        fps: fpsNow,
        etaS: (total - s.framesDone) / fpsNow,
        segment: Math.floor(s.framesDone / 96),
        segmentsTotal: Math.ceil(total / 96),
      });
      if (s.framesDone >= total) setState(active, "muxing");
    } else if (s.state === "muxing") {
      renderFps = null;
      setState(active, "done");
    }
    active = undefined;
  }, 500);

  const fail = (code: string, message: string): never => {
    throw { code, message } satisfies CmdError;
  };

  return {
    kind: "demo",
    async hardwareInfo() {
      return {
        ready: true,
        devices,
        selected: selected ?? devices[0]!.uuid,
        discoveryError: null,
        encoders: ["hevc_nvenc", "av1_nvenc", "h264_nvenc", "libsvtav1", "libkvazaar", "libopenh264"],
      };
    },
    async setPreferredDevice(uuid) {
      selected = uuid;
    },
    async probeMedia(paths) {
      return paths.map((p, i) => ({ path: p, info: fakeInfo(p, i), error: null }));
    },
    async enqueueJobs(specs) {
      return specs.map((spec, i) => {
        const id = crypto.randomUUID() as JobId;
        jobs.set(id, {
          paused: false,
          summary: { id, spec, state: "queued", queuePos: jobs.size + i, framesDone: 0, framesTotal: null, error: null },
        });
        emit({ type: "stateChanged", job: id, state: "queued", reason: null });
        return id;
      });
    },
    async listJobs() {
      return [...jobs.values()].map((s) => ({ ...s.summary }));
    },
    async pauseJob(id) {
      const s = jobs.get(id);
      if (s) {
        s.paused = true;
        setState(s, "paused", "paused");
        renderFps = null;
      }
    },
    async resumeJob(id) {
      const s = jobs.get(id);
      if (s) {
        s.paused = false;
        setState(s, s.summary.framesTotal ? "rendering" : "queued");
      }
    },
    async cancelJob(id) {
      const s = jobs.get(id);
      if (s) setState(s, "cancelled");
    },
    async removeJob(id) {
      jobs.delete(id);
    },
    async reorderJobs(order) {
      order.forEach((id, i) => {
        const s = jobs.get(id);
        if (s) s.summary.queuePos = i;
      });
    },
    async setEcoMode(mode) {
      eco = mode;
    },
    async renderPreview() {
      return fail("demo", "Previews run the real engine. Open FrameScale as the desktop app to render one.");
    },
    async thumbnail() {
      return "";
    },
    assetUrl: (rel) => rel,
    async openExternal(url) {
      window.open(url, "_blank", "noopener,noreferrer");
    },
    async openOutput() {
      fail("demo", "Opening files needs the desktop app.");
    },
    async revealOutput() {
      fail("demo", "Showing files needs the desktop app.");
    },
    async openSource() {
      fail("demo", "Opening files needs the desktop app.");
    },
    async compareOpen() {
      return fail("demo", "Comparing finished renders needs the desktop app.");
    },
    async compareClose() {},
    async pickFolder() {
      return "/Users/demo/Movies/FrameScale";
    },
    async subscribeJobs(cb) {
      jobSubs.add(cb);
      return () => jobSubs.delete(cb);
    },
    async subscribeTelemetry(cb) {
      let util = 30;
      let temp = 52;
      let vram = 3.1 * GiB;
      const h = setInterval(() => {
        const busy = renderFps != null;
        const duty = eco.mode === "fixed" ? eco.dutyPct / 100 : eco.mode === "auto" && temp > eco.targetC ? 0.6 : 1;
        util += ((busy ? 94 * duty : 6) - util) * 0.25 + (Math.random() - 0.5) * 6;
        temp += ((busy ? 58 + 22 * duty : 46) - temp) * 0.05 + (Math.random() - 0.5) * 0.6;
        vram += ((busy ? 9.2 * GiB : 1.4 * GiB) - vram) * 0.15;
        const t: Telemetry = {
          atMs: Date.now(),
          device: {
            uuid: selected ?? devices[0]!.uuid,
            utilizationPct: Math.max(0, Math.min(100, util)),
            vramUsed: vram,
            vramTotal: 12 * GiB,
            temperatureC: temp,
            powerW: busy ? 140 + Math.random() * 40 : 18 + Math.random() * 4,
          },
          renderFps: renderFps == null ? null : renderFps * duty,
          eco: { mode: eco, duty, throttling: duty < 0.999, onBattery: false },
          diskFreeBytes: 412 * GiB - (renderFps ? Date.now() % 1e6 : 0),
        };
        cb(t);
      }, 500);
      return () => clearInterval(h);
    },
    async pickFiles() {
      return ["/Users/demo/Movies/wedding_ceremony.mov", "/Users/demo/Movies/drone_coast_hdr.mov", "/Users/demo/Movies/vhs_birthday_1994.avi"];
    },
    async onFileDrop() {
      return () => {};
    },
    async showWindow() {},
  };
}
