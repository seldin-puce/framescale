import { isTauri } from "@tauri-apps/api/core";
import type { Backend } from "./backend";
import { createDemoBackend } from "./demo";
import { tauriBackend } from "./tauri";

export * from "./backend";

/** The Rust host inside Tauri; a simulation in a plain browser (`pnpm ui:dev`). */
export const backend: Backend = isTauri() ? tauriBackend : createDemoBackend();
