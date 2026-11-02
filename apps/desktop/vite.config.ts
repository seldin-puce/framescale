import { fileURLToPath, URL } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Set by the Tauri CLI during `tauri dev` / `tauri build`.
const platform = process.env.TAURI_ENV_PLATFORM;
const isDebug = !!process.env.TAURI_ENV_DEBUG;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
  },
  // Tauri expects a fixed port and must see Rust compile errors.
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_"],
  build: {
    // WebView2 is evergreen Chromium; macOS 12+ ships Safari 15+ WebKit;
    // WebKitGTK 2.40+ on supported Linux distros.
    target: platform === "windows" ? "chrome120" : "safari15",
    minify: !isDebug,
    sourcemap: isDebug,
    chunkSizeWarningLimit: 1024,
  },
});
