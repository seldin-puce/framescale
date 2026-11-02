//! `cargo run -p fs-hw --example list_devices`: print discovery + one telemetry sample.
use std::sync::Arc;

fn main() {
    match fs_hw::VulkanContext::discover() {
        Ok((ctx, mut devices)) => {
            fs_hw::rank(&mut devices);
            let ctx = Arc::new(ctx);
            for d in &devices {
                println!(
                    "{:>5}  {:<40} {:?}/{:?}  {} MiB  fp16={} coop={} sg={}..{} drm={:?} uuid={}",
                    d.score,
                    d.name,
                    d.vendor,
                    d.kind,
                    d.vram_bytes >> 20,
                    d.fp16,
                    d.cooperative_matrix,
                    d.subgroup_min,
                    d.subgroup_max,
                    d.drm_render_minor,
                    d.uuid
                );
                let mut s = fs_hw::telemetry::Sampler::for_device(d.clone(), Some(ctx.clone()));
                println!(
                    "       sources={:?} sample={:?}",
                    s.source_names(),
                    s.sample()
                );
            }
            println!(
                "default → {:?}",
                fs_hw::select_default(&devices, None).map(|d| &d.name)
            );
        }
        Err(e) => println!("discovery failed: {e}"),
    }
    println!(
        "on_battery={:?}  disk_free(/)={:?}",
        fs_hw::power::on_battery(),
        fs_hw::disk_free(std::path::Path::new("/"))
    );
}
