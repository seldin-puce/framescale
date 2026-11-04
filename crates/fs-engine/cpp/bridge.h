// fs-engine ⇄ ncnn bridge (cxx). Rust owns tiling, colour conversion and policy.
// This layer only runs networks on planar float tensors.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "rust/cxx.h"

namespace ncnn {
class Net;
}

namespace fsncnn {

struct GpuEntry;

class Model {
public:
    Model(int32_t gpu, bool fp16, int32_t threads);
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    void load(rust::Str param, rust::Str bin);
    // Same, with the param given as (possibly edited) text.
    void load_text(rust::Str param_text, rust::Str bin);
    // Param from a file, weights from memory (e.g. a blend of two checkpoints).
    // The model keeps its own copy: ncnn may reference it for its lifetime.
    void load_weights(rust::Str param, rust::Slice<const uint8_t> bin);
    // Both: param text, weights from memory.
    void load_text_weights(rust::Str param_text, rust::Slice<const uint8_t> bin);

    // Super-resolution: planar RGB [3][h][w] in 0–1 → planar [3][out_h][out_w].
    void run_sr(rust::Slice<const float> input, int32_t w, int32_t h,
                rust::Slice<float> output, int32_t out_w, int32_t out_h) const;

    // RIFE v4: two planar RGB frames (w, h multiples of 32) + timestep → frame at t.
    void run_rife(rust::Slice<const float> a, rust::Slice<const float> b,
                  int32_t w, int32_t h, float t, rust::Slice<float> output) const;

    // Real-CUGAN: the SE blocks' global means (gap0, gap1, …) of a whole
    // padded image, concatenated; `lens` receives each vector's length.
    rust::Vec<float> cugan_stats(rust::Slice<const float> input, int32_t w, int32_t h,
                                 rust::Vec<int32_t>& lens) const;

    // Real-CUGAN tile with whole-frame SE statistics injected. Tile-local means
    // would shift each tile's brightness/contrast differently (visible seams).
    void run_cugan(rust::Slice<const float> input, int32_t w, int32_t h,
                   rust::Slice<const float> gaps, rust::Slice<const int32_t> lens,
                   rust::Slice<float> output, int32_t out_w, int32_t out_h) const;

    // Detector heads: planar [3][h][w] into `in0` → outputs `out0`, `out1`, …
    // concatenated; `shapes` receives (w, h, c) per output.
    rust::Vec<float> run_outputs(rust::Slice<const float> input, int32_t w, int32_t h,
                                 rust::Vec<int32_t>& shapes) const;

    // CodeFormer (this = encoder, `generator`): planar RGB 512×512 face in
    // [-1, 1] → restored face in [-1, 1]. Top-1 codebook lookup on the CPU.
    void run_codeformer(const Model& generator, rust::Slice<const float> input,
                        rust::Slice<float> output) const;

private:
    void index_gaps();
    void load_weight_bytes(rust::Slice<const uint8_t> bin, const std::string& what);
    // A failed run: "device lost" when the GPU is gone, else ncnn's code.
    [[noreturn]] void fail_run(const char* what, int ret) const;
    // Weights loaded from memory (4-byte aligned); outlives `net_`.
    std::vector<uint32_t> weights_;
    std::unique_ptr<ncnn::Net> net_;
    int32_t gpu_;
    // Names of the SE statistics blobs (`gap0`, `gap1`, …) in index order.
    std::vector<std::string> gaps_;
};

// Create the process-wide Vulkan instance. `driver_path` may be empty (system
// loader) or an explicit ICD / MoltenVK path. Returns the GPU count, or -1.
int32_t init_gpu(rust::Str driver_path);
// Tear down the Vulkan instance. Call after every Model is dropped, before exit.
void destroy_gpu();
rust::Vec<GpuEntry> gpu_list();
std::unique_ptr<Model> model_new(int32_t gpu, bool fp16, int32_t threads);

}  // namespace fsncnn
