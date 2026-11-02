// fs-engine ⇄ ncnn bridge (cxx). Rust owns tiling, colour conversion and policy.
// This layer only runs networks on planar float tensors.
#pragma once

#include <cstdint>
#include <memory>

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

    // Super-resolution: planar RGB [3][h][w] in 0–1 → planar [3][out_h][out_w].
    void run_sr(rust::Slice<const float> input, int32_t w, int32_t h,
                rust::Slice<float> output, int32_t out_w, int32_t out_h);

    // RIFE v4: two planar RGB frames (w, h multiples of 32) + timestep → frame at t.
    void run_rife(rust::Slice<const float> a, rust::Slice<const float> b,
                  int32_t w, int32_t h, float t, rust::Slice<float> output);

private:
    std::unique_ptr<ncnn::Net> net_;
    int32_t gpu_;
};

// Create the process-wide Vulkan instance. `driver_path` may be empty (system
// loader) or an explicit ICD / MoltenVK path. Returns the GPU count, or -1.
int32_t init_gpu(rust::Str driver_path);
// Tear down the Vulkan instance. Call after every Model is dropped, before exit.
void destroy_gpu();
rust::Vec<GpuEntry> gpu_list();
std::unique_ptr<Model> model_new(int32_t gpu, bool fp16, int32_t threads);

}  // namespace fsncnn
