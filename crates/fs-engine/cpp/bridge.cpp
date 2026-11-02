#include "fs-engine/cpp/bridge.h"
#include "fs-engine/src/ncnn.rs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "gpu.h"
#include "layer.h"
#include "net.h"

namespace fsncnn {

namespace {

// ---------------------------------------------------------------------------
// rife.Warp: bilinear backward warp of `image` by optical `flow`.
// bottom[0] = image (C×H×W), bottom[1] = flow (2×H×W: dx, dy).
// CPU implementation. ncnn moves blobs between GPU and CPU around it.
// ---------------------------------------------------------------------------
class Warp : public ncnn::Layer {
public:
    Warp() { one_blob_only = false; }

    int forward(const std::vector<ncnn::Mat>& bottom, std::vector<ncnn::Mat>& top,
                const ncnn::Option& opt) const override {
        const ncnn::Mat& image = bottom[0];
        const ncnn::Mat& flow = bottom[1];
        const int w = image.w, h = image.h, c = image.c;
        ncnn::Mat& out = top[0];
        out.create(w, h, c, image.elemsize, opt.blob_allocator);
        if (out.empty()) return -100;

        const float* fx = flow.channel(0);
        const float* fy = flow.channel(1);
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++) {
            const float* src = image.channel(q);
            float* dst = out.channel(q);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    const int i = y * w + x;
                    const float sx = x + fx[i];
                    const float sy = y + fy[i];
                    const int x0 = (int)std::floor(sx), y0 = (int)std::floor(sy);
                    const float a = sx - x0, b = sy - y0;
                    const int cx0 = std::clamp(x0, 0, w - 1), cx1 = std::clamp(x0 + 1, 0, w - 1);
                    const int cy0 = std::clamp(y0, 0, h - 1), cy1 = std::clamp(y0 + 1, 0, h - 1);
                    const float v0 = src[cy0 * w + cx0], v1 = src[cy0 * w + cx1];
                    const float v2 = src[cy1 * w + cx0], v3 = src[cy1 * w + cx1];
                    dst[i] = (v0 * (1 - a) + v1 * a) * (1 - b) + (v2 * (1 - a) + v3 * a) * b;
                }
            }
        }
        return 0;
    }
};

ncnn::Layer* warp_creator(void*) { return new Warp; }

std::once_flag g_gpu_once;
int32_t g_gpu_count = -1;

[[noreturn]] void fail(const char* what, int ret) {
    // "-100" is ncnn's allocation-failure convention; Rust maps it to VRAM OOM.
    throw std::runtime_error(std::string(what) + " failed (ncnn " + std::to_string(ret) + ")");
}

ncnn::Mat wrap_planar(const float* data, int w, int h, int c) {
    // ncnn channel stride is aligned; copy plane by plane.
    ncnn::Mat m(w, h, c);
    for (int q = 0; q < c; q++) std::memcpy(m.channel(q), data + (size_t)q * w * h, sizeof(float) * w * h);
    return m;
}

void unwrap_planar(const ncnn::Mat& m, float* out, int w, int h, int c) {
    if (m.w != w || m.h != h || m.c != c)
        throw std::runtime_error("unexpected output shape " + std::to_string(m.w) + "x" + std::to_string(m.h) +
                                 "x" + std::to_string(m.c) + ", wanted " + std::to_string(w) + "x" +
                                 std::to_string(h) + "x" + std::to_string(c));
    for (int q = 0; q < c; q++) std::memcpy(out + (size_t)q * w * h, m.channel(q), sizeof(float) * w * h);
}

}  // namespace

int32_t init_gpu(rust::Str driver_path) {
    std::string path(driver_path.data(), driver_path.size());
    std::call_once(g_gpu_once, [&] {
        g_gpu_count = ncnn::create_gpu_instance(path.empty() ? nullptr : path.c_str()) == 0
                          ? ncnn::get_gpu_count()
                          : -1;
    });
    return g_gpu_count;
}

void destroy_gpu() {
    if (g_gpu_count >= 0) ncnn::destroy_gpu_instance();
    g_gpu_count = -1;
}

rust::Vec<GpuEntry> gpu_list() {
    rust::Vec<GpuEntry> out;
    if (g_gpu_count <= 0) return out;
    for (int i = 0; i < g_gpu_count; i++) {
        const ncnn::GpuInfo& info = ncnn::get_gpu_info(i);
        GpuEntry e;
        e.index = i;
        e.name = rust::String(info.device_name());
        e.vendor_id = info.vendor_id();
        e.device_id = info.device_id();
        e.kind = info.type();
        out.push_back(std::move(e));
    }
    return out;
}

Model::Model(int32_t gpu, bool fp16, int32_t threads) : net_(new ncnn::Net), gpu_(gpu) {
    ncnn::Option& o = net_->opt;
    o.num_threads = threads;
    o.use_vulkan_compute = gpu >= 0;
    o.use_fp16_packed = fp16;
    o.use_fp16_storage = fp16;
    o.use_fp16_arithmetic = false;  // fp16 accumulation visibly bands SR output
    o.use_int8_storage = false;
    o.lightmode = true;
    if (gpu >= 0) net_->set_vulkan_device(gpu);
    net_->register_custom_layer("rife.Warp", warp_creator);
}

Model::~Model() { net_->clear(); }

void Model::load(rust::Str param, rust::Str bin) {
    std::string p(param.data(), param.size()), b(bin.data(), bin.size());
    int r = net_->load_param(p.c_str());
    if (r != 0) fail(("load_param " + p).c_str(), r);
    r = net_->load_model(b.c_str());
    if (r != 0) fail(("load_model " + b).c_str(), r);
}

void Model::run_sr(rust::Slice<const float> input, int32_t w, int32_t h, rust::Slice<float> output,
                   int32_t out_w, int32_t out_h) {
    if (input.size() != (size_t)w * h * 3 || output.size() != (size_t)out_w * out_h * 3)
        throw std::runtime_error("run_sr: buffer size mismatch");
    ncnn::Extractor ex = net_->create_extractor();
    int r = ex.input("data", wrap_planar(input.data(), w, h, 3));
    if (r != 0) fail("input", r);
    ncnn::Mat out;
    r = ex.extract("output", out);
    if (r != 0) fail("extract", r);
    unwrap_planar(out, output.data(), out_w, out_h, 3);
}

void Model::run_rife(rust::Slice<const float> a, rust::Slice<const float> b, int32_t w, int32_t h, float t,
                     rust::Slice<float> output) {
    const size_t n = (size_t)w * h * 3;
    if (a.size() != n || b.size() != n || output.size() != n) throw std::runtime_error("run_rife: buffer size mismatch");
    ncnn::Mat ts(w, h, 1);
    ts.fill(t);
    ncnn::Extractor ex = net_->create_extractor();
    int r = ex.input("in0", wrap_planar(a.data(), w, h, 3));
    if (r == 0) r = ex.input("in1", wrap_planar(b.data(), w, h, 3));
    if (r == 0) r = ex.input("in2", ts);
    if (r != 0) fail("input", r);
    ncnn::Mat out;
    r = ex.extract("out0", out);
    if (r != 0) fail("extract", r);
    unwrap_planar(out, output.data(), w, h, 3);
}

std::unique_ptr<Model> model_new(int32_t gpu, bool fp16, int32_t threads) {
    if (gpu >= 0 && (g_gpu_count <= 0 || gpu >= g_gpu_count)) throw std::runtime_error("invalid GPU index");
    return std::make_unique<Model>(gpu, fp16, threads);
}

}  // namespace fsncnn
