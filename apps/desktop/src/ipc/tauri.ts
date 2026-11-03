import type { JobEvent, Telemetry } from "@framescale/ipc-types";
import { Channel, invoke } from "@tauri-apps/api/core";
import { getCurrentWebview } from "@tauri-apps/api/webview";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { open } from "@tauri-apps/plugin-dialog";
import type { Backend, HardwareInfo, ProbeResult } from "./backend";

const IS_WINDOWS = typeof navigator !== "undefined" && navigator.userAgent.includes("Windows");

const VIDEO_EXTENSIONS = ["mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "ts", "mxf", "wmv", "flv", "mpg", "mpeg"];

export const tauriBackend: Backend = {
  kind: "tauri",
  hardwareInfo: () => invoke<HardwareInfo>("hardware_info"),
  setPreferredDevice: (uuid) => invoke("set_preferred_device", { uuid }),
  probeMedia: (paths) => invoke<ProbeResult[]>("probe_media", { paths }),
  enqueueJobs: (specs) => invoke("enqueue_jobs", { specs }),
  listJobs: () => invoke("list_jobs"),
  pauseJob: (id, mode) => invoke("pause_job", { id, mode }),
  resumeJob: (id) => invoke("resume_job", { id }),
  cancelJob: (id) => invoke("cancel_job", { id }),
  removeJob: (id) => invoke("remove_job", { id }),
  reorderJobs: (order) => invoke("reorder_jobs", { order }),
  setEcoMode: (mode) => invoke("set_eco_mode", { mode }),
  renderPreview: (path, atS, seconds, pipeline) => invoke("render_preview", { path, atS, seconds, pipeline }),
  thumbnail: (path) => invoke("thumbnail", { path }),
  // Custom protocols are `scheme://localhost/…` on macOS/Linux and
  // `http://scheme.localhost/…` on Windows (WebView2).
  assetUrl: (rel) => (IS_WINDOWS ? `http://fs.localhost/${rel}` : `fs://localhost/${rel}`),

  async subscribeJobs(cb) {
    let live = true;
    const channel = new Channel<JobEvent>();
    channel.onmessage = (e) => live && cb(e);
    await invoke("subscribe_jobs", { channel });
    return () => {
      live = false;
    };
  },

  async subscribeTelemetry(cb) {
    let live = true;
    const channel = new Channel<Telemetry>();
    channel.onmessage = (t) => live && cb(t);
    await invoke("subscribe_telemetry", { channel });
    return () => {
      live = false;
    };
  },

  async pickFiles() {
    const r = await open({ multiple: true, filters: [{ name: "Video", extensions: VIDEO_EXTENSIONS }] });
    if (r == null) return [];
    return Array.isArray(r) ? r : [r];
  },

  async onFileDrop(onDrop, onHover) {
    // Tauri's native drag-drop yields real filesystem paths (HTML5 DnD can't).
    return getCurrentWebview().onDragDropEvent((e) => {
      const p = e.payload;
      if (p.type === "enter" || p.type === "over") onHover(true);
      else if (p.type === "leave") onHover(false);
      else if (p.type === "drop") {
        onHover(false);
        const videos = p.paths.filter((x) => VIDEO_EXTENSIONS.some((ext) => x.toLowerCase().endsWith(`.${ext}`)));
        if (videos.length) onDrop(videos);
      }
    });
  },

  showWindow: () => getCurrentWindow().show(),
};
