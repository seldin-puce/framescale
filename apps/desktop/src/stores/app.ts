import type {
  Container,
  EcoMode,
  Entitlements,
  Feature,
  JobEvent,
  JobId,
  JobProgress,
  JobSpec,
  JobSummary,
  MediaInfo,
  PreviewResult,
  StageConfig,
  StageKind,
  VideoCodec,
} from "@framescale/ipc-types";
import { create } from "zustand";
import { backend, isCmdError, type HardwareInfo } from "@/ipc";
import { outputPathFor } from "@/lib/format";

/* ------------------------------------------------------------------------- */
/* Pipeline model (UI-side); converted to the Rust `PipelineSpec` on submit.  */
/* ------------------------------------------------------------------------- */

export interface StageState {
  enabled: boolean;
  model: string;
  /** BYOM file for encumbered models (Face Fix). Never bundled. */
  modelPath: string | null;
  scale: 2 | 3 | 4;
  strength: 0 | 1 | 2 | 3;
  fidelity: number;
  factor: 2 | 3 | 4;
  sceneThreshold: number;
  sendField: boolean;
}

/** Stages the inference engine doesn't run yet: shown, but not selectable. */
export const COMING_SOON: ReadonlySet<StageKind> = new Set<StageKind>(["denoise", "faceFix"]);

export const STAGE_ORDER: StageKind[] = ["deinterlace", "denoise", "upscale", "faceFix", "interpolate"];

export const MODELS: Record<StageKind, { id: string; label: string }[]> = {
  deinterlace: [{ id: "bwdif", label: "BWDIF (motion adaptive)" }],
  denoise: [
    { id: "realcugan-se", label: "Real-CUGAN SE" },
    { id: "realcugan-pro", label: "Real-CUGAN Pro" },
  ],
  upscale: [
    { id: "realesrgan-x4plus", label: "Real-ESRGAN x4+ (live action)" },
    { id: "realesrgan-x4plus-anime", label: "Real-ESRGAN x4+ Anime" },
    { id: "realesr-animevideov3", label: "Real-ESR AnimeVideo v3 (fast)" },
  ],
  faceFix: [{ id: "byom", label: "Your model (CodeFormer / GFPGAN)" }],
  interpolate: [{ id: "rife-v4.6", label: "RIFE v4.6" }],
};

const defaults = (enabled: boolean, model: string): StageState => ({
  enabled,
  model,
  modelPath: null,
  scale: 2,
  strength: 1,
  fidelity: 0.7,
  factor: 2,
  sceneThreshold: 10,
  sendField: true,
});

/** Pro features a stage configuration needs (UI hint only; Rust enforces). */
export function stageFeature(kind: StageKind, s: StageState): Feature | null {
  if (kind === "faceFix") return "faceFix";
  if (kind === "interpolate" && s.factor > 2) return "interpolateHighFactor";
  return null;
}

export function toStageConfigs(stages: Record<StageKind, StageState>): StageConfig[] {
  const out: StageConfig[] = [];
  for (const kind of STAGE_ORDER) {
    const s = stages[kind];
    if (!s.enabled) continue;
    const model = { id: s.model, path: s.modelPath };
    switch (kind) {
      case "deinterlace":
        out.push({ kind, sendField: s.sendField });
        break;
      case "denoise":
        out.push({ kind, model, strength: s.strength });
        break;
      case "upscale":
        out.push({ kind, model, scale: s.scale });
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

export interface Draft {
  path: string;
  info: MediaInfo | null;
  error: string | null;
}

export interface PreviewState {
  status: "idle" | "rendering" | "ready" | "error";
  /** Source the preview belongs to (stale once the selection changes). */
  path: string | null;
  result: PreviewResult | null;
  error: string | null;
}

export type CompareMode = "split" | "side" | "before" | "after";

export interface OutputState {
  codec: VideoCodec;
  container: Container;
  quality: number;
}

interface AppStore {
  hardware: HardwareInfo | null;
  entitlements: Entitlements;
  developmentKey: boolean;
  drafts: Draft[];
  jobs: Record<JobId, JobSummary>;
  progress: Record<JobId, JobProgress>;
  warnings: Record<JobId, string[]>;
  selected: string | null; // job id or draft path
  stages: Record<StageKind, StageState>;
  selectedStage: StageKind;
  output: OutputState;
  eco: EcoMode;
  compareMode: CompareMode;
  proModal: { open: boolean; feature: Feature | null };
  toast: { kind: "error" | "info"; message: string } | null;
  preview: PreviewState;

  init(): Promise<void>;
  renderPreview(atS: number): Promise<void>;
  selectedSource(): { path: string; info: MediaInfo | null } | null;
  addFiles(paths: string[]): Promise<void>;
  removeDraft(path: string): void;
  renderAll(): Promise<void>;
  select(id: string | null): void;
  setStage(kind: StageKind, patch: Partial<StageState>): void;
  selectStage(kind: StageKind): void;
  setOutput(patch: Partial<OutputState>): void;
  setEco(mode: EcoMode): Promise<void>;
  setCompareMode(m: CompareMode): void;
  openPro(feature: Feature | null): void;
  closePro(): void;
  activate(token: string): Promise<string | null>;
  setDevice(uuid: string | null): Promise<void>;
  jobAction(id: JobId, action: "pause" | "hardPause" | "resume" | "cancel" | "remove"): Promise<void>;
  handleEvent(e: JobEvent): void;
  has(f: Feature): boolean;
  fail(e: unknown): void;
}

export const useApp = create<AppStore>()((set, get) => ({
  hardware: null,
  entitlements: { tier: "free", features: [], licensee: null, expiresAt: null },
  developmentKey: false,
  drafts: [],
  jobs: {},
  progress: {},
  warnings: {},
  selected: null,
  stages: {
    deinterlace: defaults(false, "bwdif"),
    denoise: defaults(false, "realcugan-se"),
    upscale: defaults(true, "realesrgan-x4plus"),
    faceFix: defaults(false, "byom"),
    interpolate: defaults(false, "rife-v4.6"),
  },
  selectedStage: "upscale",
  output: { codec: "hevc10", container: "mp4", quality: 20 },
  eco: { mode: "off" },
  compareMode: "split",
  proModal: { open: false, feature: null },
  toast: null,
  preview: { status: "idle", path: null, result: null, error: null },

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
    set({ preview: { status: "rendering", path: src.path, result: null, error: null } });
    try {
      const result = await backend.renderPreview(src.path, atS, { stages: toStageConfigs(get().stages) });
      // Ignore a result for a clip that's no longer selected.
      if (get().preview.path === src.path) set({ preview: { status: "ready", path: src.path, result, error: null } });
    } catch (e) {
      const message = isCmdError(e) ? e.message : String(e);
      set({ preview: { status: "error", path: src.path, result: null, error: message } });
      get().fail(e);
    }
  },

  async init() {
    const [license, jobs] = await Promise.all([backend.getLicense(), backend.listJobs()]);
    set({
      entitlements: license.entitlements,
      developmentKey: license.developmentKey,
      jobs: Object.fromEntries(jobs.map((j) => [j.id, j])),
    });
    // Hardware discovery runs in the background on the Rust side; poll until ready.
    const poll = async () => {
      const hw = await backend.hardwareInfo();
      set({ hardware: hw });
      if (!hw.ready) setTimeout(poll, 500);
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
        path: outputPathFor(d.path, scale, output.container),
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
  setOutput: (patch) => set((s) => ({ output: { ...s.output, ...patch } })),
  async setEco(mode) {
    set({ eco: mode });
    await backend.setEcoMode(mode);
  },
  setCompareMode: (m) => set({ compareMode: m }),
  openPro: (feature) => set({ proModal: { open: true, feature } }),
  closePro: () => set({ proModal: { open: false, feature: null } }),

  async activate(token) {
    try {
      const l = await backend.activateLicense(token);
      set({ entitlements: l.entitlements, developmentKey: l.developmentKey });
      return null;
    } catch (e) {
      return isCmdError(e) ? e.message : String(e);
    }
  },

  async setDevice(uuid) {
    await backend.setPreferredDevice(uuid);
    set({ hardware: await backend.hardwareInfo() });
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
            jobs: job ? { ...s.jobs, [e.job]: { ...job, framesDone: e.framesDone, framesTotal: e.framesTotal } } : s.jobs,
          };
        case "stateChanged":
          if (!job) {
            void backend.listJobs().then((js) => set({ jobs: Object.fromEntries(js.map((j) => [j.id, j])) }));
            return {};
          }
          return { jobs: { ...s.jobs, [e.job]: { ...job, state: e.state, error: e.reason } } };
        case "warning":
          return { warnings: { ...s.warnings, [e.job]: [...(s.warnings[e.job] ?? []), e.message] } };
        case "error":
          return { toast: { kind: "error", message: e.message } };
      }
    });
  },

  has: (f) => get().entitlements.features.includes(f),

  fail(e) {
    if (isCmdError(e) && e.code === "proRequired") {
      set({ proModal: { open: true, feature: e.feature } });
    } else {
      set({ toast: { kind: "error", message: isCmdError(e) ? e.message : String(e) } });
    }
  },
}));
