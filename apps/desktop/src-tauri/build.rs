fn main() {
    // Phase 3: switch to `tauri_build::try_build(Attributes::new().app_manifest(
    // AppManifest::new().commands(&[...])))` so every app command needs an
    // explicit grant in capabilities/main.json (deny-by-default IPC).
    tauri_build::build();
}
