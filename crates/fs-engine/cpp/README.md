# fs-engine C++ shim

Phase 3 adds the `cxx` bridge (`bridge.h` / `bridge.cpp`) here. It wraps
`ncnn::Net`, `ncnn::VkCompute` and the custom u16 ⇄ fp16 conversion shaders.

`build.rs` builds ncnn from the pinned full-source archive (see
`third_party/manifest.toml`), with these settings:

- `NCNN_VULKAN=ON`, `NCNN_SIMPLEVK=ON` (dlopen the Vulkan loader / MoltenVK at runtime)
- `NCNN_BUILD_TOOLS=OFF`, `NCNN_BUILD_EXAMPLES=OFF`, `NCNN_BUILD_BENCHMARK=OFF`
- static library, `-fvisibility=hidden`

Set `NCNN_DIR=/path/to/prebuilt` to skip the source build during local iteration.
