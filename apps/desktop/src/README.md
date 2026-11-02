# Frontend source layout

| Dir           | Contents (Phase 3) |
|---------------|--------------------|
| `app/`        | `App`, providers, `AppShell` |
| `components/` | `shell/`, `queue/`, `viewport/`, `inspector/`, `telemetry/`, `pro/`, `ui/` (primitives) |
| `stores/`     | Zustand slices: `queue`, `pipeline`, `preview`, `license`, `ui` |
| `ipc/`        | Typed `invoke` wrappers + Channel subscriptions (types from `@framescale/ipc-types`) |
| `gl/`         | WebGL2 compare renderer (split/side/toggle shaders, loupe) |
| `hooks/`      | `useTelemetry` (external ring buffer → `useSyncExternalStore`), etc. |
| `lib/`        | `cn()`, formatters (bytes, fps, ETA, timecode) |
| `styles/`     | `theme.css`: design tokens |

The UI holds no secrets and never spawns processes. Pro gating here is
cosmetic; enforcement happens in Rust (core + fs-worker).
