import type {
  CompareInfo,
  DeviceUuid,
  EcoMode,
  GpuDevice,
  JobEvent,
  JobId,
  JobSpec,
  JobSummary,
  MediaInfo,
  PauseMode,
  PipelineSpec,
  PreviewResult,
  Telemetry,
} from "@framescale/ipc-types";

export interface HardwareInfo {
  ready: boolean;
  devices: GpuDevice[];
  selected: DeviceUuid | null;
  discoveryError: string | null;
  encoders: string[];
}

export interface ProbeResult {
  path: string;
  info: MediaInfo | null;
  error: string | null;
}


/** Mirrors `commands::CmdError` in the Rust host. */
export interface CmdError {
  code: string;
  message: string;
}

export function isCmdError(e: unknown): e is CmdError {
  return typeof e === "object" && e !== null && "code" in e && "message" in e;
}

export type Unsubscribe = () => void;

/**
 * Everything the UI may ask of the host. The Tauri implementation talks to
 * Rust; the demo implementation simulates it for browser-only development.
 */
export interface Backend {
  readonly kind: "tauri" | "demo";
  hardwareInfo(): Promise<HardwareInfo>;
  setPreferredDevice(uuid: DeviceUuid | null): Promise<void>;
  probeMedia(paths: string[]): Promise<ProbeResult[]>;
  enqueueJobs(specs: JobSpec[]): Promise<JobId[]>;
  listJobs(): Promise<JobSummary[]>;
  pauseJob(id: JobId, mode: PauseMode): Promise<void>;
  resumeJob(id: JobId): Promise<void>;
  cancelJob(id: JobId): Promise<void>;
  removeJob(id: JobId): Promise<void>;
  reorderJobs(order: JobId[]): Promise<void>;
  setEcoMode(mode: EcoMode): Promise<void>;
  /** Render `seconds` (0 = one frame, max 2) of `path` from `atS` through `pipeline` (real engine). */
  renderPreview(path: string, atS: number, seconds: number, pipeline: PipelineSpec): Promise<PreviewResult>;
  /** Cached JPEG thumbnail; resolves to an `fs://` path for `assetUrl`. */
  thumbnail(path: string): Promise<string>;
  /** URL for an `fs://` path (preview frames, thumbnails). */
  assetUrl(rel: string): string;
  /** Open an external link in the system browser (Ko-fi / GitHub only). */
  openExternal(url: string): Promise<void>;
  /** Play a finished job's output in the system player. */
  openOutput(id: JobId): Promise<void>;
  /** Show a finished job's output, selected, in the file manager. */
  revealOutput(id: JobId): Promise<void>;
  /** Play a job's original source in the system player. */
  openSource(id: JobId): Promise<void>;
  /** Start a source-vs-output comparison of a finished job. */
  compareOpen(id: JobId, maxWidth: number): Promise<CompareInfo>;
  /** Stop a comparison's decoders (frames: `compare/<session>/<side>/[full/]<n>.jpg`). */
  compareClose(session: string): Promise<void>;
  /** Folder picker for the output destination; null if cancelled. */
  pickFolder(): Promise<string | null>;
  /** Rejects with a message unless `path` holds a CodeFormer-ncnn model. */
  checkFaceModel(path: string): Promise<void>;
  subscribeJobs(cb: (e: JobEvent) => void): Promise<Unsubscribe>;
  subscribeTelemetry(cb: (t: Telemetry) => void): Promise<Unsubscribe>;
  pickFiles(): Promise<string[]>;
  onFileDrop(onDrop: (paths: string[]) => void, onHover: (over: boolean) => void): Promise<Unsubscribe>;
  showWindow(): Promise<void>;
}
