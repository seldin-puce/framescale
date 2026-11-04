import type {
  Container,
  EcoMode,
  JobEvent,
  JobId,
  JobProgress,
  JobSpec,
  JobSummary,
  MediaInfo,
  ModelStatus,
  DownloadProgress,
  PreviewProgress,
  PreviewResult,
  StageConfig,
  StageKind,
  VideoCodec,
} from "@framescale/ipc-types";
import { create } from "zustand";
import { backend, isCmdError, type HardwareInfo } from "@/ipc";
import { outputPathFor } from "@/lib/format";
import { supportNudge } from "@/lib/support";

/* ------------------------------------------------------------------------- */
/* Pipeline model (UI-side); converted to the Rust `PipelineSpec` on submit.  */
/* ------------------------------------------------------------------------- */

export interface StageState {
  enabled: boolean;
  model: string;
  scale: 2 | 4;
  /** Upscale with Real-ESRGAN General v3: noise removal 0–1 (blends its two checkpoints). */
  denoise: number;
  strength: 0 | 1 | 2 | 3;
  fidelity: number;
  factor: 2 | 3 | 4;
  sceneThreshold: number;
  sendField: boolean;
}

/** Stages the inference engine doesn't run yet: shown, but not selectable. */
/** Stages shown but not runnable yet (none right now). */
export const COMING_SOON: ReadonlySet<StageKind> = new Set<StageKind>();

/** Mirrors `Container::accepts` in fs-types: what FFmpeg can mux. */
export function containerAccepts(container: Container, codec: VideoCodec): boolean {
  if (container === "mov") return codec !== "av1";
  if (container === "mp4") return codec !== "proRes422Hq" && codec !== "proRes4444";
  return true;
}

/** Upscale model with a noise-removal blend. */
export const GENERAL_V3 = "realesr-general-x4v3";

export const STAGE_ORDER: StageKind[] = ["deinterlace", "denoise", "upscale", "faceFix", "interpolate"];

export const MODELS: Record<StageKind, { id: string; label: string }[]> = {
  deinterlace: [{ id: "bwdif", label: "BWDIF (motion adaptive)" }],
  denoise: [
    { id: "nafnet-sidd", label: "NAFNet (live action, camera noise)" },
    { id: "realcugan-se", label: "Real-CUGAN SE (4 levels)" },
    { id: "realcugan-pro", label: "Real-CUGAN Pro (finer detail)" },
  ],
  upscale: [
    { id: "realesr-general-x4v3", label: "Real-ESRGAN General v3 (live action, fast)" },
    // ×2 runs RealESRGAN_x2plus, ×3/×4 run x4plus (fs-engine resolve_sr).
    { id: "realesrgan-x4plus", label: "Real-ESRGAN x4+/x2+ (live action, slow)" },
    { id: "realesrgan-x4plus-anime", label: "Real-ESRGAN x4+ Anime" },
    { id: "realesr-animevideov3", label: "Real-ESR AnimeVideo v3 (fast)" },
  ],
  faceFix: [{ id: "codeformer", label: "CodeFormer" }],
  interpolate: [
    { id: "rife-v4.25", label: "RIFE 4.25 (recommended)" },
    { id: "rife-v4.26", label: "RIFE 4.26 (newest)" },
    { id: "rife-v4.22-lite", label: "RIFE 4.22 lite (lighter)" },
    { id: "rife-v4.6", label: "RIFE 4.6 (classic)" },
  ],
};

/** On-demand model id of Face Fix's CodeFormer (third_party/manifest.toml). */
export const FACE_MODEL = "codeformer";

export interface FaceModelState {
  /** Null until the first check. */
  status: ModelStatus | null;
  /** Set while downloading. */
  progress: DownloadProgress | null;
  error: string | null;
}

const defaults = (enabled: boolean, model: string): StageState => ({
  enabled,
  model,
  scale: 2,
  denoise: 0.5,
  strength: 1,
  fidelity: 0.7,
  factor: 2,
  sceneThreshold: 10,
  sendField: true,
});

export function toStageConfigs(stages: Record<StageKind, StageState>): StageConfig[] {
  const out: StageConfig[] = [];
  for (const kind of STAGE_ORDER) {
    const s = stages[kind];
    if (!s.enabled) continue;
    const model = { id: s.model, path: null };
    switch (kind) {
      case "deinterlace":
        out.push({ kind, sendField: s.sendField });
        break;
      case "denoise":
        out.push({ kind, model, strength: s.strength });
        break;
      case "upscale":
        out.push({ kind, model, scale: s.scale, denoise: s.model === GENERAL_V3 ? s.denoise : null });
        break;
      case "faceFix":
        out.push({ kind, model, fidelity: s.fidelity });
        break;
      case "interpolate":
        out.push({ kind, model, factor: s.factor, sceneThreshold: s.sceneThreshold });
        break;
    }
  }
  return out;
}

/* ------------------------------------------------------------------------- */

/** Encoders per codec family, hardware first (mirrors fs-media::encoders). */
export const CODEC_ENCODERS: Record<"hevc10" | "av1" | "h264", string[]> = {
  hevc10: ["hevc_nvenc", "hevc_amf", "hevc_qsv", "hevc_vaapi", "hevc_videotoolbox", "libx265", "libkvazaar"],
  av1: ["av1_nvenc", "av1_amf", "av1_qsv", "av1_vaapi", "libsvtav1"],
  h264: ["h264_nvenc", "h264_amf", "h264_qsv", "h264_vaapi", "h264_videotoolbox", "libx264", "libopenh264"],
};

export const hasHardware = (codec: keyof typeof CODEC_ENCODERS, usable: string[]) =>
  CODEC_ENCODERS[codec].some((e) => !e.startsWith("lib") && usable.includes(e));

/**
 * Best default: a hardware encoder if there is one (HEVC first, for player
 * compatibility), else x265 HEVC (plays everywhere), else SVT-AV1.
 */
export function preferredCodec(usable: string[]): VideoCodec {
  if (hasHardware("hevc10", usable)) return "hevc10";
  if (hasHardware("av1", usable)) return "av1";
  if (usable.includes("libx265")) return "hevc10";
  return "av1";
}

export interface Draft {
  path: string;
  info: MediaInfo | null;
  error: string | null;
}

/** A preview render in flight: first a single frame, then the full 2 s. */
export interface PreviewRun extends PreviewProgress {
  phase: "first" | "full";
}

export interface PreviewState {
  status: "idle" | "rendering" | "ready" | "error";
  /** Source the preview belongs to (stale once the selection changes). */
  path: string | null;
  result: PreviewResult | null;
  error: string | null;
  /** A single-frame still is shown while the full 2 s renders. */
  refining: boolean;
  /** Progress of the render in flight; null when none runs. */
  run: PreviewRun | null;
}

let previewToken = 0;

const OUTPUT_DIR_KEY = "framescale.outputDir";

function loadOutputDir(): string | null {
  try {
    return localStorage.getItem(OUTPUT_DIR_KEY);
  } catch {
    return null;
  }
}

export type CompareMode = "split" | "side" | "before" | "after";

export interface OutputState {
  codec: VideoCodec;
  container: Container;
  quality: number;
}

interface AppStore {
  hardware: HardwareInfo | null;
  drafts: Draft[];
  jobs: Record<JobId, JobSummary>;
  progress: Record<JobId, JobProgress>;
  warnings: Record<JobId, string[]>;
  selected: string | null; // job id or draft path
  stages: Record<StageKind, StageState>;
  selectedStage: StageKind;
  output: OutputState;
  /** The user picked output settings themselves: never auto-change them. */
  outputTouched: boolean;
  eco: EcoMode;
  compareMode: CompareMode;
  /** About / support dialog. */
  aboutOpen: boolean;
  /** The occasional, dismissible "Enjoying FrameScale?" line after a render. */
  nudge: boolean;
  /** Most recently finished job: drives the completion card and status bar. */
  completed: JobId | null;
  /** Card dismissed (the status bar keeps the link). */
  completedDismissed: boolean;
  /** Output folder; null = next to each source file. */
  outputDir: string | null;
  /** Start in-app comparison playback once this job's comparison is open. */
  autoplay: JobId | null;
  toast: { kind: "error" | "info"; message: string } | null;
  preview: PreviewState;
  /** CodeFormer (Face Fix) isn't in the installer: downloaded on first use. */
  faceModel: FaceModelState;

  init(): Promise<void>;
  renderPreview(atS: number): Promise<void>;
  /** Stop the preview render in flight (a first frame that already arrived stays). */
  cancelPreview(): void;
  selectedSource(): { path: string; info: MediaInfo | null } | null;
  addFiles(paths: string[]): Promise<void>;
  removeDraft(path: string): void;
  renderAll(): Promise<void>;
  select(id: string | null): void;
  setStage(kind: StageKind, patch: Partial<StageState>): void;
  /** The stage's on/off switch. Face Fix needs CodeFormer downloaded first. */
  toggleStage(kind: StageKind): void;
  selectStage(kind: StageKind): void;
  downloadFaceModel(): Promise<void>;
  cancelFaceModelDownload(): void;
  setOutput(patch: Partial<OutputState>): void;
  setEco(mode: EcoMode): Promise<void>;
  setCompareMode(m: CompareMode): void;
  setAboutOpen(open: boolean): void;
  dismissNudge(): void;
  dismissCompleted(): void;
  setOutputDir(dir: string | null): void;
  /** Open / reveal / compare actions for finished jobs. */
  openOutput(id: JobId): Promise<void>;
  revealOutput(id: JobId): Promise<void>;
  openSource(id: JobId): Promise<void>;
  /** Show a finished job and play it against its source, in the app. */
  playCompare(id: JobId): void;
  clearAutoplay(): void;
  setDevice(uuid: string | null): Promise<void>;
  jobAction(id: JobId, action: "pause" | "hardPause" | "resume" | "cancel" | "remove"): Promise<void>;
  /** Stop a job now and put its clip back in "Ready to render". Asks first
   *  when more than a minute of rendering would be thrown away. */
  requestStop(id: JobId): void;
  /** The job a stop is waiting to be confirmed for. */
  confirmStop: JobId | null;
  setConfirmStop(id: JobId | null): void;
  stopJob(id: JobId): Promise<void>;
  /** Jobs being stopped → their input, to return to the drafts once cancelled. */
  stopping: Record<JobId, string>;
  /** When each job's first progress report arrived (ms), for the stop prompt. */
  renderStartedAt: Record<JobId, number>;
  handleEvent(e: JobEvent): void;
  fail(e: unknown): void;
}

export const useApp = create<AppStore>()((set, get) => ({
  hardware: null,
  drafts: [],
  jobs: {},
  progress: {},
  confirmStop: null,
  stopping: {},
  renderStartedAt: {},
  warnings: {},
  selected: null,
  stages: {
    deinterlace: defaults(false, "bwdif"),
    denoise: defaults(false, "nafnet-sidd"),
    upscale: defaults(true, GENERAL_V3),
    faceFix: defaults(false, "codeformer"),
    interpolate: defaults(false, "rife-v4.25"),
  },
  selectedStage: "upscale",
  output: { codec: "hevc10", container: "mp4", quality: 20 },
  outputTouched: false,
  eco: { mode: "off" },
  compareMode: "split",
  aboutOpen: false,
  nudge: false,
  completed: null,
  completedDismissed: false,
  outputDir: loadOutputDir(),
  autoplay: null,
  toast: null,
  preview: { status: "idle", path: null, result: null, error: null, refining: false, run: null },
  faceModel: { status: null, progress: null, error: null },

  selectedSource() {
    const { selected, drafts, jobs } = get();
    if (!selected) return null;
    const d = drafts.find((x) => x.path === selected);
    if (d) return { path: d.path, info: d.info };
    const j = jobs[selected as JobId];
    return j ? { path: j.spec.input, info: null } : null;
  },

  async renderPreview(atS) {
    const src = get().selectedSource();
    if (!src) return;
    // Heavy models take ~1 s/frame, so 2 s of video can take tens of seconds.
    // Show one frame almost immediately, then swap in the full clip.
    const token = ++previewToken;
    const pipeline = { stages: toStageConfigs(get().stages) };
    const current = () => token === previewToken && get().preview.path === src.path;
    const report = (phase: PreviewRun["phase"]) => (p: PreviewProgress) => {
      if (current()) set((s) => ({ preview: { ...s.preview, run: { ...p, phase } } }));
    };
    const waiting = (phase: PreviewRun["phase"]): PreviewRun => ({ phase, started: false, done: 0, total: 0, fps: 0 });
    set({ preview: { status: "rendering", path: src.path, result: null, error: null, refining: false, run: waiting("first") } });
    try {
      const still = await backend.renderPreview(src.path, atS, 0, pipeline, report("first"));
      if (!current()) return;
      set({ preview: { status: "ready", path: src.path, result: still, error: null, refining: true, run: waiting("full") } });
      const full = await backend.renderPreview(src.path, atS, 2, pipeline, report("full"));
      if (!current()) return;
      set({ preview: { status: "ready", path: src.path, result: full, error: null, refining: false, run: null } });
    } catch (e) {
      if (!current()) return;
      const keep = get().preview.result;
      if (isCmdError(e) && e.code === "cancelled") {
        set({ preview: { status: keep ? "ready" : "idle", path: src.path, result: keep, error: null, refining: false, run: null } });
        return;
      }
      const message = isCmdError(e) ? e.message : String(e);
      set({ preview: { status: keep ? "ready" : "error", path: src.path, result: keep, error: message, refining: false, run: null } });
      get().fail(e);
    }
  },

  cancelPreview() {
    const p = get().preview;
    if (!p.run) return;
    previewToken++; // the pending render resolves into nothing
    set({ preview: { ...p, status: p.result ? "ready" : "idle", refining: false, run: null } });
    void backend.cancelPreview();
  },

  async init() {
    const jobs = await backend.listJobs();
    set({ jobs: Object.fromEntries(jobs.map((j) => [j.id, j])) });
    void backend
      .modelStatus(FACE_MODEL)
      .then((status) => set((s) => ({ faceModel: { ...s.faceModel, status } })))
      .catch(() => {});
    // Hardware discovery runs in the background on the Rust side; poll until ready.
    const poll = async () => {
      const hw = await backend.hardwareInfo();
      set({ hardware: hw });
      if (!hw.ready) {
        setTimeout(poll, 500);
        return;
      }
      // Don't leave a default the machine can't encode in hardware (e.g. Fedora's
      // Mesa ships VAAPI without H.264/HEVC, but AV1 works on RDNA3+).
      const { output, outputTouched } = get();
      if (!outputTouched && output.codec !== preferredCodec(hw.encoders)) {
        const codec = preferredCodec(hw.encoders);
        const container = containerAccepts(output.container, codec) ? output.container : codec === "av1" ? "mp4" : "mov";
        set({ output: { ...output, codec, container } });
      }
    };
    void poll();
  },

  async addFiles(paths) {
    const known = new Set(get().drafts.map((d) => d.path));
    const fresh = paths.filter((p) => !known.has(p));
    if (!fresh.length) return;
    set((s) => ({ drafts: [...s.drafts, ...fresh.map((path) => ({ path, info: null, error: null }))] }));
    const results = await backend.probeMedia(fresh);
    set((s) => ({
      drafts: s.drafts.map((d) => {
        const r = results.find((x) => x.path === d.path);
        return r ? { ...d, info: r.info, error: r.error } : d;
      }),
      selected: s.selected ?? fresh[0] ?? null,
    }));
  },

  removeDraft(path) {
    set((s) => ({ drafts: s.drafts.filter((d) => d.path !== path) }));
  },

  async renderAll() {
    const { drafts, stages, output } = get();
    const ready = drafts.filter((d) => d.info);
    if (!ready.length) return;
    const pipeline = { stages: toStageConfigs(stages) };
    const scale = stages.upscale.enabled ? stages.upscale.scale : 1;
    const specs: JobSpec[] = ready.map((d) => ({
      input: d.path,
      pipeline,
      output: {
        path: outputPathFor(d.path, scale, output.container, get().outputDir),
        codec: output.codec,
        container: output.container,
        quality: output.quality,
        encoder: "auto",
      },
      device: null,
    }));
    try {
      await backend.enqueueJobs(specs);
      const jobs = await backend.listJobs();
      set({
        drafts: drafts.filter((d) => !d.info),
        jobs: Object.fromEntries(jobs.map((j) => [j.id, j])),
      });
    } catch (e) {
      get().fail(e);
    }
  },

  select: (id) => set({ selected: id }),
  setStage: (kind, patch) => set((s) => ({ stages: { ...s.stages, [kind]: { ...s.stages[kind], ...patch } } })),
  selectStage: (kind) => set({ selectedStage: kind }),
  toggleStage: (kind) => {
    const s = get().stages[kind];
    // Face Fix without CodeFormer: show its download panel instead.
    if (kind === "faceFix" && !s.enabled && !get().faceModel.status?.installed) {
      set({ selectedStage: kind });
      return;
    }
    get().setStage(kind, { enabled: !s.enabled });
  },

  async downloadFaceModel() {
    if (get().faceModel.progress) return;
    const start: DownloadProgress = { done: 0, total: get().faceModel.status?.bytes ?? 0 };
    set((s) => ({ faceModel: { ...s.faceModel, progress: start, error: null } }));
    try {
      await backend.downloadModel(FACE_MODEL, (progress) => set((s) => ({ faceModel: { ...s.faceModel, progress } })));
      const status = await backend.modelStatus(FACE_MODEL);
      set((s) => ({
        faceModel: { status, progress: null, error: null },
        stages: status.installed ? { ...s.stages, faceFix: { ...s.stages.faceFix, enabled: true } } : s.stages,
      }));
    } catch (e) {
      const cancelled = isCmdError(e) && e.code === "cancelled";
      const message = isCmdError(e) ? e.message : String(e);
      set((s) => ({ faceModel: { ...s.faceModel, progress: null, error: cancelled ? null : message } }));
    }
  },

  cancelFaceModelDownload: () => void backend.cancelModelDownload(),
  setOutput: (patch) =>
    set((s) => {
      const next = { ...s.output, ...patch };
      // Keep codec and container compatible: a new codec moves the container.
      if (!containerAccepts(next.container, next.codec)) {
        next.container = patch.container && !patch.codec ? next.container : next.codec === "av1" ? "mp4" : "mov";
        if (!containerAccepts(next.container, next.codec)) next.container = "mkv";
      }
      return { output: next, outputTouched: true };
    }),
  async setEco(mode) {
    set({ eco: mode });
    await backend.setEcoMode(mode);
  },
  setCompareMode: (m) => set({ compareMode: m }),
  setAboutOpen: (open) => set({ aboutOpen: open }),
  dismissNudge: () => {
    supportNudge.dismissed();
    set({ nudge: false });
  },
  dismissCompleted: () => set({ completedDismissed: true, nudge: false }),
  playCompare: (id) => set({ selected: id, autoplay: id, completedDismissed: true }),
  clearAutoplay: () => set({ autoplay: null }),
  setOutputDir: (dir) => {
    try {
      if (dir) localStorage.setItem(OUTPUT_DIR_KEY, dir);
      else localStorage.removeItem(OUTPUT_DIR_KEY);
    } catch {
      // storage unavailable: the choice lasts for this session
    }
    set({ outputDir: dir });
  },
  async openOutput(id) {
    try {
      await backend.openOutput(id);
    } catch (e) {
      get().fail(e);
    }
  },
  async revealOutput(id) {
    try {
      await backend.revealOutput(id);
    } catch (e) {
      get().fail(e);
    }
  },
  async openSource(id) {
    try {
      await backend.openSource(id);
    } catch (e) {
      get().fail(e);
    }
  },

  async setDevice(uuid) {
    await backend.setPreferredDevice(uuid);
    set({ hardware: await backend.hardwareInfo() });
  },

  requestStop(id) {
    const started = get().renderStartedAt[id];
    if (started != null && Date.now() - started > STOP_CONFIRM_AFTER_MS) set({ confirmStop: id });
    else void get().stopJob(id);
  },

  setConfirmStop: (id) => set({ confirmStop: id }),

  async stopJob(id) {
    const job = get().jobs[id];
    if (!job) return;
    set((s) => ({ confirmStop: null, stopping: { ...s.stopping, [id]: job.spec.input } }));
    try {
      // Immediate: the worker drops the segment in progress (no checkpoint).
      await backend.cancelJob(id);
    } catch (e) {
      set((s) => {
        const { [id]: _, ...stopping } = s.stopping;
        return { stopping };
      });
      get().fail(e);
    }
  },

  async jobAction(id, action) {
    try {
      if (action === "pause") await backend.pauseJob(id, "soft");
      else if (action === "hardPause") await backend.pauseJob(id, "hard");
      else if (action === "resume") await backend.resumeJob(id);
      else if (action === "cancel") await backend.cancelJob(id);
      else {
        await backend.removeJob(id);
        set((s) => {
          const { [id]: _, ...jobs } = s.jobs;
          return { jobs, selected: s.selected === id ? null : s.selected };
        });
      }
    } catch (e) {
      get().fail(e);
    }
  },

  handleEvent(e) {
    set((s) => {
      const job = s.jobs[e.job];
      switch (e.type) {
        case "progress":
          return {
            progress: { ...s.progress, [e.job]: e },
            renderStartedAt: s.renderStartedAt[e.job] != null ? s.renderStartedAt : { ...s.renderStartedAt, [e.job]: Date.now() },
            jobs: job ? { ...s.jobs, [e.job]: { ...job, framesDone: e.framesDone, framesTotal: e.framesTotal } } : s.jobs,
          };
        case "stateChanged":
          if (!job) {
            void backend.listJobs().then((js) => set({ jobs: Object.fromEntries(js.map((j) => [j.id, j])) }));
            return {};
          }
          return {
            jobs: { ...s.jobs, [e.job]: { ...job, state: e.state, error: e.reason } },
            ...(e.state === "done" ? { completed: e.job, completedDismissed: false } : {}),
            // A finished render is the one moment a gentle "tip jar" note is welcome.
            nudge: s.nudge || (e.state === "done" && supportNudge.due()),
          };
        case "warning":
          return { warnings: { ...s.warnings, [e.job]: [...(s.warnings[e.job] ?? []), e.message] } };
        case "error":
          return { toast: { kind: "error", message: e.message } };
      }
    });
    // A stopped job leaves the list and its clip goes back to "Ready to render",
    // so the user can change the pipeline and render it again.
    if (e.type === "stateChanged" && e.state === "cancelled") {
      const input = get().stopping[e.job];
      if (input != null) void returnToDrafts(e.job, input);
    }
  },

  fail(e) {
    set({ toast: { kind: "error", message: isCmdError(e) ? e.message : String(e) } });
  },
}));

/** Rendering this long (ms) before a stop asks for confirmation. */
const STOP_CONFIRM_AFTER_MS = 60_000;

async function returnToDrafts(id: JobId, input: string) {
  const { getState, setState } = useApp;
  // The queue releases a job just before it reports it cancelled; retry
  // briefly in case removal races that.
  for (let attempt = 0; ; attempt++) {
    try {
      await backend.removeJob(id);
      break;
    } catch (e) {
      if (attempt >= 5 || !(isCmdError(e) && e.code === "running")) {
        getState().fail(e);
        break;
      }
      await new Promise((r) => setTimeout(r, 200));
    }
  }
  setState((s) => {
    const { [id]: _j, ...jobs } = s.jobs;
    const { [id]: _p, ...progress } = s.progress;
    const { [id]: _s, ...stopping } = s.stopping;
    const { [id]: _r, ...renderStartedAt } = s.renderStartedAt;
    return { jobs, progress, stopping, renderStartedAt, selected: s.selected === id ? null : s.selected };
  });
  await getState().addFiles([input]);
}
