//! Builds ncnn from the pinned source (feature `ncnn`) and the cxx bridge.
//!
//! Source: `$NCNN_SRC`, else `<workspace>/.cache/ncnn/src` (`cargo xtask ncnn`).
//! Prebuilt: set `NCNN_DIR` to an install prefix (include/ncnn + lib) to skip
//! the source build during local iteration.

fn main() {
    // `ncnn_omp`: ncnn runs layers multi-threaded itself (OpenMP). Without it
    // (MSVC), the backend parallelises CPU work across tiles instead.
    println!("cargo::rustc-check-cfg=cfg(ncnn_omp)");
    #[cfg(feature = "ncnn")]
    ncnn::build();
}

#[cfg(feature = "ncnn")]
mod ncnn {
    use std::path::{Path, PathBuf};

    pub fn build() {
        println!("cargo:rerun-if-changed=cpp/bridge.cpp");
        println!("cargo:rerun-if-changed=cpp/bridge.h");
        println!("cargo:rerun-if-changed=src/ncnn.rs");
        println!("cargo:rerun-if-env-changed=NCNN_SRC");
        println!("cargo:rerun-if-env-changed=NCNN_DIR");

        if !msvc() {
            // Source builds enable ncnn's simpleomp there; prebuilt NCNN_DIR
            // installs are assumed to match.
            println!("cargo:rustc-cfg=ncnn_omp");
        }
        let prefix = match std::env::var_os("NCNN_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => build_from_source(),
        };
        let include = prefix.join("include").join("ncnn");
        let lib_dir = ["lib", "lib64"]
            .iter()
            .map(|l| prefix.join(l))
            .find(|p| p.join(libname("ncnn")).exists())
            .unwrap_or_else(|| panic!("libncnn not found under {}", prefix.display()));

        let mut bridge = cxx_build::bridge("src/ncnn.rs");
        bridge
            .file("cpp/bridge.cpp")
            .include(&include)
            .std("c++17")
            .flag_if_supported("-fvisibility=hidden")
            .flag_if_supported("-Wno-unknown-pragmas")
            .flag_if_supported("/EHsc");
        // The Warp layer's `omp parallel for` resolves against ncnn's simpleomp.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
            bridge.flag("-Xclang").flag("-fopenmp");
        } else if !msvc() {
            bridge.flag_if_supported("-fopenmp");
        }
        bridge.compile("fs_ncnn_bridge");

        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        // ncnn first, then glslang's libraries (order matters for GNU ld).
        println!("cargo:rustc-link-lib=static=ncnn");
        let mut extra: Vec<String> = std::fs::read_dir(&lib_dir)
            .into_iter()
            .flatten()
            .flatten()
            .filter_map(|e| {
                let n = e.file_name().to_string_lossy().into_owned();
                let stem = n.strip_prefix("lib").unwrap_or(&n);
                let stem = stem
                    .strip_suffix(".a")
                    .or_else(|| stem.strip_suffix(".lib"))?;
                (stem != "ncnn").then(|| stem.to_owned())
            })
            .collect();
        // glslang → (MachineIndependent → OSDependent, GenericCodeGen) → SPIRV ordering.
        let rank = |s: &str| match s {
            "SPIRV" => 0,
            "glslang" => 1,
            "MachineIndependent" => 2,
            "OSDependent" => 3,
            "GenericCodeGen" => 4,
            _ => 5,
        };
        extra.sort_by_key(|s| (rank(s), s.clone()));
        for l in extra {
            println!("cargo:rustc-link-lib=static={l}");
        }
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        match target_os.as_str() {
            "linux" => {
                println!("cargo:rustc-link-lib=dylib=stdc++");
                println!("cargo:rustc-link-lib=dylib=dl");
                println!("cargo:rustc-link-lib=dylib=pthread");
            }
            "macos" => println!("cargo:rustc-link-lib=dylib=c++"),
            _ => {}
        }
    }

    fn msvc() -> bool {
        std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
    }

    fn libname(stem: &str) -> String {
        if msvc() {
            format!("{stem}.lib")
        } else {
            format!("lib{stem}.a")
        }
    }

    fn build_from_source() -> PathBuf {
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
        let src = std::env::var_os("NCNN_SRC")
            .map(PathBuf::from)
            .unwrap_or_else(|| manifest.join("../../.cache/ncnn/src"));
        if !Path::new(&src).join("CMakeLists.txt").exists() {
            panic!(
                "ncnn source not found at {}. Run `cargo xtask ncnn` (or set NCNN_SRC / NCNN_DIR).",
                src.display()
            );
        }
        let off = |c: &mut cmake::Config, k: &str| {
            c.define(k, "OFF");
        };
        let mut cfg = cmake::Config::new(&src);
        // Always optimise ncnn: a debug build is ~20× slower and useless for testing.
        cfg.profile("Release")
            .define("NCNN_VULKAN", "ON")
            .define("NCNN_SIMPLEVK", "ON") // dlopen the loader/MoltenVK at runtime; no SDK needed
            .define("NCNN_INSTALL_SDK", "ON");
        for k in [
            "NCNN_SHARED_LIB",
            "NCNN_BUILD_TOOLS",
            "NCNN_BUILD_EXAMPLES",
            "NCNN_BUILD_BENCHMARK",
            "NCNN_BUILD_TESTS",
            "NCNN_PYTHON",
            "NCNN_SYSTEM_GLSLANG",
        ] {
            off(&mut cfg, k);
        }
        if msvc() {
            // Match Rust's +crt-static (.cargo/config.toml). MSVC's OpenMP needs
            // vcomp140.dll (plus the dynamic CRT), which a static-CRT app doesn't
            // ship. ncnn stays single-threaded per call; the backend runs one
            // tile per core instead (see `ncnn_omp`).
            cfg.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded");
            off(&mut cfg, "NCNN_OPENMP");
        } else {
            // ncnn's self-contained OpenMP runtime: multi-threaded CPU path with
            // no libgomp/libomp to bundle or sign.
            cfg.define("NCNN_OPENMP", "ON")
                .define("NCNN_SIMPLEOMP", "ON");
        }
        cfg.build()
    }
}
