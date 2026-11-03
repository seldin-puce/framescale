#include "fs-engine/cpp/bridge.h"
#include "fs-engine/src/ncnn.rs.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include "command.h"
#include "gpu.h"
#include "layer.h"
#include "net.h"
#include "pipeline.h"

namespace fsncnn {

namespace {

// ---------------------------------------------------------------------------
// rife.Warp: bilinear backward warp of `image` by optical `flow`, with border
// clamping (PyTorch grid_sample, padding_mode="border").
// bottom[0] = image (C×H×W), bottom[1] = flow (2×H×W: dx, dy).
// Runs on the GPU in Vulkan nets (after nihui/rife-ncnn-vulkan's warp, MIT);
// the CPU version serves CPU nets. Coordinates stay fp32 in the shader: fp16
// can't address pixels past ~2048 precisely.
// ---------------------------------------------------------------------------
const char* const kWarpPack1 = R"glsl(
#version 450
layout(binding = 0) readonly buffer image_blob { sfp image_data[]; };
layout(binding = 1) readonly buffer flow_blob { sfp flow_data[]; };
layout(binding = 2) writeonly buffer top_blob { sfp top_data[]; };
layout(push_constant) uniform parameter { int w; int h; int c; int cstep; int fcstep; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= p.w || gy >= p.h || gz >= p.c) return;
    int i = gy * p.w + gx;
    float sx = float(gx) + float(buffer_ld1(flow_data, i));
    float sy = float(gy) + float(buffer_ld1(flow_data, p.fcstep + i));
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float a = sx - float(x0);
    float b = sy - float(y0);
    int cx0 = clamp(x0, 0, p.w - 1);
    int cx1 = clamp(x0 + 1, 0, p.w - 1);
    int cy0 = clamp(y0, 0, p.h - 1);
    int cy1 = clamp(y0 + 1, 0, p.h - 1);
    int base = gz * p.cstep;
    float v0 = float(buffer_ld1(image_data, base + cy0 * p.w + cx0));
    float v1 = float(buffer_ld1(image_data, base + cy0 * p.w + cx1));
    float v2 = float(buffer_ld1(image_data, base + cy1 * p.w + cx0));
    float v3 = float(buffer_ld1(image_data, base + cy1 * p.w + cx1));
    float v = mix(mix(v0, v1, a), mix(v2, v3, a), b);
    buffer_st1(top_data, base + i, afp(v));
}
)glsl";

const char* const kWarpPack4 = R"glsl(
#version 450
layout(binding = 0) readonly buffer image_blob { sfpvec4 image_data[]; };
layout(binding = 1) readonly buffer flow_blob { sfp flow_data[]; };
layout(binding = 2) writeonly buffer top_blob { sfpvec4 top_data[]; };
layout(push_constant) uniform parameter { int w; int h; int c; int cstep; int fcstep; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    int gz = int(gl_GlobalInvocationID.z);
    if (gx >= p.w || gy >= p.h || gz >= p.c) return;
    int i = gy * p.w + gx;
    float sx = float(gx) + float(buffer_ld1(flow_data, i));
    float sy = float(gy) + float(buffer_ld1(flow_data, p.fcstep + i));
    int x0 = int(floor(sx));
    int y0 = int(floor(sy));
    float a = sx - float(x0);
    float b = sy - float(y0);
    int cx0 = clamp(x0, 0, p.w - 1);
    int cx1 = clamp(x0 + 1, 0, p.w - 1);
    int cy0 = clamp(y0, 0, p.h - 1);
    int cy1 = clamp(y0 + 1, 0, p.h - 1);
    int base = gz * p.cstep;
    vec4 v0 = vec4(buffer_ld4(image_data, base + cy0 * p.w + cx0));
    vec4 v1 = vec4(buffer_ld4(image_data, base + cy0 * p.w + cx1));
    vec4 v2 = vec4(buffer_ld4(image_data, base + cy1 * p.w + cx0));
    vec4 v3 = vec4(buffer_ld4(image_data, base + cy1 * p.w + cx1));
    vec4 v = mix(mix(v0, v1, a), mix(v2, v3, a), b);
    buffer_st4(top_data, base + i, afpvec4(v));
}
)glsl";

class Warp : public ncnn::Layer {
public:
    Warp() {
        one_blob_only = false;
        support_vulkan = true;
        support_vulkan_packing = true;
    }

    int create_pipeline(const ncnn::Option& opt) override {
        if (!vkdev) return 0;
        for (int pack : {1, 4}) {
            const char* src = pack == 4 ? kWarpPack4 : kWarpPack1;
            std::vector<uint32_t> spirv;
            int r = ncnn::compile_spirv_module(src, (int)std::strlen(src), opt, spirv);
            if (r != 0) return r;
            auto* pipeline = new ncnn::Pipeline(vkdev);
            pipeline->set_optimal_local_size_xyz(8, 8, pack == 4 ? 1 : 4);
            r = pipeline->create(spirv.data(), spirv.size() * 4, {});
            if (r != 0) {
                delete pipeline;
                return r;
            }
            (pack == 4 ? pack4_ : pack1_) = pipeline;
        }
        return 0;
    }

    int destroy_pipeline(const ncnn::Option&) override {
        delete pack1_;
        delete pack4_;
        pack1_ = pack4_ = nullptr;
        return 0;
    }

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

    int forward(const std::vector<ncnn::VkMat>& bottom, std::vector<ncnn::VkMat>& top, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override {
        const ncnn::VkMat& image = bottom[0];
        const ncnn::VkMat& flow = bottom[1];
        if (flow.elempack != 1 || flow.c < 2 || flow.w != image.w || flow.h != image.h) return -1;
        ncnn::VkMat& out = top[0];
        out.create(image.w, image.h, image.c, image.elemsize, image.elempack, opt.blob_vkallocator);
        if (out.empty()) return -100;
        std::vector<ncnn::VkMat> bindings{image, flow, out};
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = out.w;
        constants[1].i = out.h;
        constants[2].i = out.c;
        constants[3].i = (int)out.cstep;
        constants[4].i = (int)flow.cstep;
        cmd.record_pipeline(image.elempack == 4 ? pack4_ : pack1_, bindings, constants, out);
        return 0;
    }

private:
    ncnn::Pipeline* pack1_ = nullptr;
    ncnn::Pipeline* pack4_ = nullptr;
};
ncnn::Layer* warp_creator(void*) { return new Warp; }

// ---------------------------------------------------------------------------
// fs.ChannelNorm: LayerNorm across channels at every pixel (NAFNet's
// LayerNorm2d, affine folded into the next 1×1 conv). One GPU thread per pixel
// keeps mean and variance in fp32 registers: no squared map is stored (fp16
// would overflow), and the dispatch is 2-D, not one workgroup per pixel
// (ncnn's Reduction tops out at 65535 outputs per dispatch row).
// Param 0: eps (default 1e-6).
// ---------------------------------------------------------------------------
const char* const kChannelNormPack1 = R"glsl(
#version 450
layout(binding = 0) readonly buffer bottom_blob { sfp bottom_data[]; };
layout(binding = 1) writeonly buffer top_blob { sfp top_data[]; };
layout(push_constant) uniform parameter { int w; int h; int c; int cstep; float eps; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    if (gx >= p.w || gy >= p.h) return;
    int i = gy * p.w + gx;
    float sum = 0.0;
    for (int q = 0; q < p.c; q++) sum += float(buffer_ld1(bottom_data, q * p.cstep + i));
    float mean = sum / float(p.c);
    float sq = 0.0;
    for (int q = 0; q < p.c; q++) {
        float d = float(buffer_ld1(bottom_data, q * p.cstep + i)) - mean;
        sq += d * d;
    }
    float inv = inversesqrt(sq / float(p.c) + p.eps);
    for (int q = 0; q < p.c; q++) {
        float d = float(buffer_ld1(bottom_data, q * p.cstep + i)) - mean;
        buffer_st1(top_data, q * p.cstep + i, afp(d * inv));
    }
}
)glsl";

const char* const kChannelNormPack4 = R"glsl(
#version 450
layout(binding = 0) readonly buffer bottom_blob { sfpvec4 bottom_data[]; };
layout(binding = 1) writeonly buffer top_blob { sfpvec4 top_data[]; };
layout(push_constant) uniform parameter { int w; int h; int c; int cstep; float eps; } p;
void main()
{
    int gx = int(gl_GlobalInvocationID.x);
    int gy = int(gl_GlobalInvocationID.y);
    if (gx >= p.w || gy >= p.h) return;
    int i = gy * p.w + gx;
    vec4 sum = vec4(0.0);
    for (int q = 0; q < p.c; q++) sum += vec4(buffer_ld4(bottom_data, q * p.cstep + i));
    float mean = (sum.x + sum.y + sum.z + sum.w) / float(p.c * 4);
    float sq = 0.0;
    for (int q = 0; q < p.c; q++) {
        vec4 d = vec4(buffer_ld4(bottom_data, q * p.cstep + i)) - mean;
        sq += dot(d, d);
    }
    float inv = inversesqrt(sq / float(p.c * 4) + p.eps);
    for (int q = 0; q < p.c; q++) {
        vec4 d = vec4(buffer_ld4(bottom_data, q * p.cstep + i)) - mean;
        buffer_st4(top_data, q * p.cstep + i, afpvec4(d * inv));
    }
}
)glsl";

class ChannelNorm : public ncnn::Layer {
public:
    ChannelNorm() {
        one_blob_only = true;
        support_vulkan = true;
        support_vulkan_packing = true;
    }

    int load_param(const ncnn::ParamDict& pd) override {
        eps_ = pd.get(0, 1e-6f);
        return 0;
    }

    int create_pipeline(const ncnn::Option& opt) override {
        if (!vkdev) return 0;
        for (int pack : {1, 4}) {
            const char* src = pack == 4 ? kChannelNormPack4 : kChannelNormPack1;
            std::vector<uint32_t> spirv;
            int r = ncnn::compile_spirv_module(src, (int)std::strlen(src), opt, spirv);
            if (r != 0) return r;
            auto* pipeline = new ncnn::Pipeline(vkdev);
            pipeline->set_optimal_local_size_xyz(8, 8, 1);
            r = pipeline->create(spirv.data(), spirv.size() * 4, {});
            if (r != 0) {
                delete pipeline;
                return r;
            }
            (pack == 4 ? pack4_ : pack1_) = pipeline;
        }
        return 0;
    }

    int destroy_pipeline(const ncnn::Option&) override {
        delete pack1_;
        delete pack4_;
        pack1_ = pack4_ = nullptr;
        return 0;
    }

    int forward(const ncnn::Mat& bottom, ncnn::Mat& top, const ncnn::Option& opt) const override {
        const int w = bottom.w, h = bottom.h, c = bottom.c;
        top.create(w, h, c, bottom.elemsize, opt.blob_allocator);
        if (top.empty()) return -100;
        const int size = w * h;
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < size; i++) {
            float sum = 0.f;
            for (int q = 0; q < c; q++) sum += bottom.channel(q)[i];
            const float mean = sum / c;
            float sq = 0.f;
            for (int q = 0; q < c; q++) {
                const float d = bottom.channel(q)[i] - mean;
                sq += d * d;
            }
            const float inv = 1.f / std::sqrt(sq / c + eps_);
            for (int q = 0; q < c; q++) top.channel(q)[i] = (bottom.channel(q)[i] - mean) * inv;
        }
        return 0;
    }

    int forward(const ncnn::VkMat& bottom, ncnn::VkMat& top, ncnn::VkCompute& cmd,
                const ncnn::Option& opt) const override {
        top.create(bottom.w, bottom.h, bottom.c, bottom.elemsize, bottom.elempack, opt.blob_vkallocator);
        if (top.empty()) return -100;
        std::vector<ncnn::VkMat> bindings{bottom, top};
        std::vector<ncnn::vk_constant_type> constants(5);
        constants[0].i = top.w;
        constants[1].i = top.h;
        constants[2].i = top.c;
        constants[3].i = (int)top.cstep;
        constants[4].f = eps_;
        ncnn::VkMat dispatcher;
        dispatcher.w = top.w;
        dispatcher.h = top.h;
        dispatcher.c = 1;
        cmd.record_pipeline(bottom.elempack == 4 ? pack4_ : pack1_, bindings, constants, dispatcher);
        return 0;
    }

private:
    float eps_ = 1e-6f;
    ncnn::Pipeline* pack1_ = nullptr;
    ncnn::Pipeline* pack4_ = nullptr;
};

ncnn::Layer* channel_norm_creator(void*) { return new ChannelNorm; }

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
    // FP16 *arithmetic* measured on RX 9070 XT / Mesa 26.2: SR got slower
    // (matrix cores already run the convolutions in FP16) and max error reached
    // 231/255 (visible artifacts). Storage stays FP16; math stays FP32.
    o.use_fp16_arithmetic = false;
    o.use_int8_storage = false;
    o.lightmode = true;
    if (gpu >= 0) net_->set_vulkan_device(gpu);
    net_->register_custom_layer("rife.Warp", warp_creator);
    net_->register_custom_layer("fs.ChannelNorm", channel_norm_creator);
}

Model::~Model() { net_->clear(); }

void Model::load(rust::Str param, rust::Str bin) {
    std::string p(param.data(), param.size()), b(bin.data(), bin.size());
    int r = net_->load_param(p.c_str());
    if (r != 0) fail(("load_param " + p).c_str(), r);
    r = net_->load_model(b.c_str());
    if (r != 0) fail(("load_model " + b).c_str(), r);
    index_gaps();
}

void Model::load_weights(rust::Str param, rust::Slice<const uint8_t> bin) {
    std::string p(param.data(), param.size());
    int r = net_->load_param(p.c_str());
    if (r != 0) fail(("load_param " + p).c_str(), r);
    weights_.assign((bin.size() + 3) / 4, 0);
    std::memcpy(weights_.data(), bin.data(), bin.size());
    const size_t used = net_->load_model(reinterpret_cast<const unsigned char*>(weights_.data()));
    if (used != bin.size())
        throw std::runtime_error("load_model: used " + std::to_string(used) + " of " + std::to_string(bin.size()) +
                                 " weight bytes (" + p + ")");
}

void Model::load_text(rust::Str param_text, rust::Str bin) {
    std::string p(param_text.data(), param_text.size()), b(bin.data(), bin.size());
    int r = net_->load_param_mem(p.c_str());
    if (r != 0) fail("load_param (text)", r);
    r = net_->load_model(b.c_str());
    if (r != 0) fail(("load_model " + b).c_str(), r);
    index_gaps();
}

void Model::index_gaps() {
    // Real-CUGAN marks its SE global-average outputs `gap<N>`.
    std::vector<std::pair<int, std::string>> found;
    for (const ncnn::Blob& blob : net_->blobs()) {
        const std::string& n = blob.name;
        if (n.size() > 3 && n.compare(0, 3, "gap") == 0 &&
            std::all_of(n.begin() + 3, n.end(), [](unsigned char ch) { return std::isdigit(ch) != 0; }))
            found.emplace_back(std::stoi(n.substr(3)), n);
    }
    std::sort(found.begin(), found.end());
    gaps_.clear();
    for (auto& f : found) gaps_.push_back(f.second);
}

rust::Vec<float> Model::cugan_stats(rust::Slice<const float> input, int32_t w, int32_t h,
                                    rust::Vec<int32_t>& lens) const {
    if (input.size() != (size_t)w * h * 3) throw std::runtime_error("cugan_stats: buffer size mismatch");
    if (gaps_.empty()) throw std::runtime_error("cugan_stats: model has no SE statistics (gap*) blobs");
    ncnn::Extractor ex = net_->create_extractor();
    int r = ex.input("in0", wrap_planar(input.data(), w, h, 3));
    if (r != 0) fail("input", r);
    rust::Vec<float> out;
    lens.clear();
    for (const std::string& g : gaps_) {
        ncnn::Mat m;
        r = ex.extract(g.c_str(), m);
        if (r != 0) fail(("extract " + g).c_str(), r);
        if (m.dims != 1)
            throw std::runtime_error("cugan_stats: " + g + " has " + std::to_string(m.dims) + " dims, expected 1");
        const float* p = static_cast<const float*>(m.data);
        for (int i = 0; i < m.w; i++) out.push_back(p[i]);
        lens.push_back(m.w);
    }
    return out;
}

rust::Vec<float> Model::run_outputs(rust::Slice<const float> input, int32_t w, int32_t h,
                                    rust::Vec<int32_t>& shapes) const {
    if (input.size() != (size_t)w * h * 3) throw std::runtime_error("run_outputs: buffer size mismatch");
    ncnn::Extractor ex = net_->create_extractor();
    int r = ex.input("in0", wrap_planar(input.data(), w, h, 3));
    if (r != 0) fail("input", r);
    rust::Vec<float> out;
    shapes.clear();
    const std::vector<ncnn::Blob>& blobs = net_->blobs();
    for (int i = 0;; i++) {
        const std::string name = "out" + std::to_string(i);
        int index = -1;
        for (size_t b = 0; b < blobs.size(); b++)
            if (blobs[b].name == name) {
                index = (int)b;
                break;
            }
        if (index < 0) break;
        ncnn::Mat m;
        r = ex.extract(index, m);
        if (r != 0) fail(("extract " + name).c_str(), r);
        for (int q = 0; q < m.c; q++) {
            const float* p = m.channel(q);
            for (int k = 0; k < m.w * m.h; k++) out.push_back(p[k]);
        }
        shapes.push_back(m.w);
        shapes.push_back(m.h);
        shapes.push_back(m.c);
    }
    if (shapes.empty()) throw std::runtime_error("run_outputs: model has no out0 blob");
    return out;
}

void Model::run_codeformer(const Model& generator, rust::Slice<const float> input,
                           rust::Slice<float> output) const {
    const int S = 512;
    if (input.size() != (size_t)S * S * 3 || output.size() != (size_t)S * S * 3)
        throw std::runtime_error("run_codeformer: buffer size mismatch");
    static const char* feats[4] = {"enc_feat_32", "enc_feat_64", "enc_feat_128", "enc_feat_256"};
    ncnn::Extractor enc = net_->create_extractor();
    int r = enc.input("input", wrap_planar(input.data(), S, S, 3));
    if (r != 0) fail("encoder input", r);
    ncnn::Mat f[4], lq, probs;
    for (int i = 0; i < 4 && r == 0; i++) r = enc.extract(feats[i], f[i]);
    if (r == 0) r = enc.extract("lq_feat", lq);
    if (r == 0) r = enc.extract("soft_one_hot", probs);
    if (r != 0) fail("encoder extract", r);
    if (probs.dims != 2) throw std::runtime_error("run_codeformer: unexpected code logits shape");
    // Each token takes its most likely codebook entry (one-hot rows).
    ncnn::Mat onehot(probs.w, probs.h);
    onehot.fill(0.f);
    for (int y = 0; y < probs.h; y++) {
        const float* row = probs.row(y);
        onehot.row(y)[std::max_element(row, row + probs.w) - row] = 1.f;
    }
    ncnn::Extractor gen = generator.net_->create_extractor();
    for (int i = 0; i < 4 && r == 0; i++) r = gen.input(feats[i], f[i]);
    if (r == 0) r = gen.input("style_feat", lq);
    if (r == 0) r = gen.input("input", onehot);
    if (r != 0) fail("generator input", r);
    if (generator.net_->output_indexes().empty()) throw std::runtime_error("run_codeformer: generator has no output");
    ncnn::Mat out;
    r = gen.extract(generator.net_->output_indexes()[0], out);
    if (r != 0) fail("generator extract", r);
    unwrap_planar(out, output.data(), S, S, 3);
}

void Model::run_cugan(rust::Slice<const float> input, int32_t w, int32_t h, rust::Slice<const float> gaps,
                      rust::Slice<const int32_t> lens, rust::Slice<float> output, int32_t out_w,
                      int32_t out_h) const {
    if (input.size() != (size_t)w * h * 3 || output.size() != (size_t)out_w * out_h * 3)
        throw std::runtime_error("run_cugan: buffer size mismatch");
    if (lens.size() != gaps_.size()) throw std::runtime_error("run_cugan: SE statistics don't match the model");
    ncnn::Extractor ex = net_->create_extractor();
    int r = ex.input("in0", wrap_planar(input.data(), w, h, 3));
    size_t off = 0;
    for (size_t i = 0; i < gaps_.size() && r == 0; i++) {
        if (lens[i] <= 0 || off + (size_t)lens[i] > gaps.size()) throw std::runtime_error("run_cugan: bad SE statistics");
        ncnn::Mat g(lens[i]);
        std::memcpy(g.data, gaps.data() + off, sizeof(float) * (size_t)lens[i]);
        off += (size_t)lens[i];
        r = ex.input(gaps_[i].c_str(), g);  // replaces the tile-local global pooling
    }
    if (r != 0) fail("input", r);
    ncnn::Mat out;
    r = ex.extract("out0", out);
    if (r != 0) fail("extract", r);
    unwrap_planar(out, output.data(), out_w, out_h, 3);
}

void Model::run_sr(rust::Slice<const float> input, int32_t w, int32_t h, rust::Slice<float> output,
                   int32_t out_w, int32_t out_h) const {
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
                     rust::Slice<float> output) const {
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
