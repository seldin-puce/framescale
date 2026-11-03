import type {
  DeviceUuid,
  EcoMode,
  Entitlements,
  Feature,
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

export interface LicenseInfo {
  entitlements: Entitlements;
  developmentKey: boolean;
}

/** Mirrors `commands::CmdError` in the Rust host. */
export interface CmdError {
  code: string;
  message: string;
  feature: Feature | null;
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
  getLicense(): Promise<LicenseInfo>;
  activateLicense(token: string): Promise<LicenseInfo>;
  setEcoMode(mode: EcoMode): Promise<void>;
  /** Render 2 s of `path` from `atS` through `pipeline` (real engine). */
  renderPreview(path: string, atS: number, pipeline: PipelineSpec): Promise<PreviewResult>;
  /** Cached JPEG thumbnail; resolves to an `fs://` path for `assetUrl`. */
  thumbnail(path: string): Promise<string>;
  /** URL for an `fs://` path (preview frames, thumbnails). */
  assetUrl(rel: string): string;
  subscribeJobs(cb: (e: JobEvent) => void): Promise<Unsubscribe>;
  subscribeTelemetry(cb: (t: Telemetry) => void): Promise<Unsubscribe>;
  pickFiles(): Promise<string[]>;
  onFileDrop(onDrop: (paths: string[]) => void, onHover: (over: boolean) => void): Promise<Unsubscribe>;
  showWindow(): Promise<void>;
}
