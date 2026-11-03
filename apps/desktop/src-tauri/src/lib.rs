//! FrameScale AI desktop host.
//!
//! Phase 2 wires up the shell only: plugins, the main window with
//! platform-native materials, and logging. Phase 3 adds commands, Channels,
//! the `fs://` preview protocol, and the supervisor.
#![deny(unsafe_code)]

mod commands;
mod paths;
mod protocol;
mod queue;
mod settings;
mod state;
mod telemetry;

use tauri::{Manager, WebviewWindowBuilder};

/// Label of the single main window; must match `app.windows[0].label`
/// in tauri.conf.json and `windows` in capabilities/main.json.
const MAIN_WINDOW: &str = "main";

pub fn run() {
    init_tracing();

    tauri::Builder::default()
        // Must be registered first. A second launch (e.g. "Open with…") focuses
        // the running instance instead of starting a second supervisor/reaper.
        .plugin(tauri_plugin_single_instance::init(|app, _argv, _cwd| {
            // Phase 3: forward media paths from `_argv` into the queue.
            if let Some(w) = app.get_webview_window(MAIN_WINDOW) {
                let _ = w.unminimize();
                let _ = w.set_focus();
            }
        }))
        .plugin(tauri_plugin_window_state::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        // Opens donation / source links in the system browser (scoped in capabilities).
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .register_asynchronous_uri_scheme_protocol("fs", |ctx, request, responder| {
            let app = ctx.app_handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = app
                    .try_state::<std::sync::Arc<state::AppState>>()
                    .map(|s| s.inner().clone());
                let response = match state {
                    Some(state) => protocol::serve(state, request).await,
                    None => tauri::http::Response::builder()
                        .status(503)
                        .body(Vec::new())
                        .unwrap_or_default(),
                };
                responder.respond(response);
            });
        })
        .setup(|app| {
            // Windows: tie every descendant process to our lifetime (zombie reaper).
            if let Err(e) = fs_proc::install_kill_on_close_job() {
                tracing::error!(error = %e, "could not create the kill-on-close job object");
            }
            let paths = paths::Paths::resolve(app.handle())?;
            let state = state::AppState::init(paths)
                .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
            app.manage(state.clone());
            tauri::async_runtime::spawn(state.clone().init_background());
            tauri::async_runtime::spawn(queue::run(app.handle().clone(), state.clone()));
            tauri::async_runtime::spawn(telemetry::run(state));
            create_main_window(app)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::hardware_info,
            commands::set_preferred_device,
            commands::probe_media,
            commands::enqueue_jobs,
            commands::list_jobs,
            commands::pause_job,
            commands::resume_job,
            commands::cancel_job,
            commands::remove_job,
            commands::reorder_jobs,
            commands::set_eco_mode,
            commands::disk_free,
            commands::render_preview,
            commands::thumbnail,
            commands::open_output,
            commands::reveal_output,
            commands::open_source,
            commands::compare_open,
            commands::compare_close,
            commands::subscribe_jobs,
            commands::subscribe_telemetry,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            tracing::error!(error = %e, "fatal: tauri runtime failed");
            std::process::exit(1);
        });
}

/// The window is declared in tauri.conf.json with `create: false` and built
/// here, so per-OS materials live in one place instead of being duplicated
/// across platform config files (JSON Merge Patch replaces arrays wholesale).
fn create_main_window(app: &mut tauri::App) -> tauri::Result<()> {
    let config = app
        .config()
        .app
        .windows
        .iter()
        .find(|w| w.label == MAIN_WINDOW)
        .cloned()
        .ok_or_else(|| tauri::Error::WindowNotFound)?;

    #[allow(unused_mut)]
    let mut builder = WebviewWindowBuilder::from_config(app.handle(), &config)?;

    #[cfg(target_os = "macos")]
    {
        use tauri::utils::config::WindowEffectsConfig;
        use tauri::window::{Effect, EffectState};
        builder = builder
            .title_bar_style(tauri::TitleBarStyle::Overlay)
            .hidden_title(true)
            .transparent(true)
            .effects(WindowEffectsConfig {
                effects: vec![Effect::Sidebar],
                state: Some(EffectState::FollowsWindowActiveState),
                radius: None,
                color: None,
            });
    }

    #[cfg(windows)]
    {
        use tauri::utils::config::WindowEffectsConfig;
        use tauri::window::Effect;
        // Mica on Windows 11. Windows 10 ignores the effect, and the solid
        // CSS background shows through instead.
        builder = builder.transparent(true).effects(WindowEffectsConfig {
            effects: vec![Effect::Mica],
            state: None,
            radius: None,
            color: None,
        });
    }

    // Linux/WebKitGTK: opaque window. Compositor blur is inconsistent there
    // and CSS backdrop-filter is expensive, so the UI uses solid surfaces.

    // Starts hidden (`visible: false`); the UI calls `show()` after first paint,
    // so the user never sees a white flash.
    builder.build()?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{EnvFilter, fmt};
    let filter = EnvFilter::try_from_env("FRAMESCALE_LOG")
        .unwrap_or_else(|_| EnvFilter::new("info,framescale=debug"));
    let _ = fmt().with_env_filter(filter).with_target(false).try_init();
}
