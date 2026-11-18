// T-709 (Docs/design/50-c-abi.md §50.af, Docs/tasks/T-709.md item 4): world layers through the C ABI on an
// offscreen 1x target -- xt_surface_set_world_layers, render_layers_gpu, world_layer_count / _info /
// _readback, draw_layers_gpu, push_input_ray -- compared with the C++ path as `xtframe-view --layers` runs it
// (gpu::WorldLayerBuilder through its LayerHost, the same frames: update(0), then 1/60 s steps). The test
// runs xtframe-view itself into its temp root (XT_CAPI_TEST_VIEW_EXE; the cases that need it are skipped
// with a message in a build without the tool) and reads its PNGs (straight alpha, re-premultiplied here)
// and layers.json. Fixtures: Examples/demos/world-layers (six layers: #Rings redrawn every frame, #Tags
// moving, #Fore fading, the tilted #Panel with the ENGAGE button) and Tests/capi/fixtures/layers (two
// coplanar card layers over a base layer; the `dot` port toggles a mark inside #right).
// Built only with xt_wgpu (exported with XTFrame-Backends, 45 §45.8); without an adapter the cases SKIP.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <json.hpp>
#include <string>
#include <vector>

#include "../common/test_temp.h"
#include "doctest.h"
#include "xt/capi/xtframe.h"
#include "xt/sw/sw_png.h"

#ifdef XT_CAPI_TEST_HAS_WGPU

namespace {

using Pixels = std::vector<uint8_t>;

bool contains(const char* text, const char* part) {
    return text != nullptr && std::strstr(text, part) != nullptr;
}

constexpr int kDemoW = 1280, kDemoH = 720;
constexpr int kDemoFrames = 30;
constexpr float kYaw = -30.0f, kPitch = 14.0f, kDepth = 1.0f;

/// An offscreen 1x target, or null (the case is skipped) without a WebGPU adapter.
xt_gpu_target* makeTarget(int width, int height) {
    xt_gpu_target_desc desc;
    std::memset(&desc, 0, sizeof desc);
    desc.struct_size = sizeof desc;
    desc.window.kind = XT_WINDOW_NONE;
    desc.width = width;
    desc.height = height;
    desc.vsync = 1;
    desc.backend = "";
    desc.sample_count = 1;  // xtframe-view --msaa 1
    xt_gpu_target* t = xt_gpu_target_create(&desc);
    if (t == nullptr) {
        const std::string error = xt_capi_last_error();
        REQUIRE_MESSAGE(contains(error.c_str(), "no WebGPU adapter"), error);
        std::printf("SKIP: no WebGPU adapter (%s)\n", error.c_str());
    }
    return t;
}

/// The bundle, surfaces and targets of one case, destroyed at scope end (surfaces, targets, bundles).
struct Fixture {
    std::vector<xt_bundle*> bundles;
    std::vector<xt_surface*> surfaces;
    std::vector<xt_gpu_target*> targets;
    ~Fixture() {
        for (xt_surface* s : surfaces) xt_surface_destroy(s);
        for (xt_gpu_target* t : targets) (void)xt_gpu_target_destroy(t);
        for (xt_bundle* b : bundles) (void)xt_bundle_close(b);
    }
    xt_surface* surface(const char* dir, const char* ui, int width, int height) {
        xt_bundle* b = xt_bundle_open(dir);
        REQUIRE_MESSAGE(b != nullptr, xt_capi_last_error());
        bundles.push_back(b);
        xt_surface_desc desc{};
        desc.struct_size = sizeof desc;
        desc.width = width;
        desc.height = height;
        desc.device_pixel_ratio = 1.0f;
        desc.premultiplied = 1;
        desc.system_fonts = 1;  // xtframe-view: the bundle's fonts, then the system's
        xt_surface* s = xt_surface_create(b, ui, &desc);
        REQUIRE_MESSAGE(s != nullptr, xt_capi_last_error());
        surfaces.push_back(s);
        return s;
    }
    xt_gpu_target* target(int width, int height) {
        xt_gpu_target* t = makeTarget(width, height);
        if (t != nullptr) targets.push_back(t);
        return t;
    }
};

/// One frame: update(dt), then render_layers_gpu (and the draw, when `m` is given) in a frame of `t`.
void layerFrame(xt_surface* s, xt_gpu_target* t, float dt, const float* m = nullptr, float depth = kDepth) {
    REQUIRE(xt_surface_update(s, dt) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_begin_frame(t, nullptr) == XT_STATUS_OK);
    REQUIRE_MESSAGE(xt_surface_render_layers_gpu(s, t) == XT_STATUS_OK, xt_capi_last_error());
    if (m != nullptr)
        REQUIRE_MESSAGE(xt_surface_draw_layers_gpu(s, t, nullptr, m, depth) == XT_STATUS_OK,
                        xt_capi_last_error());
    REQUIRE(xt_gpu_target_end_frame(t) == XT_STATUS_OK);
}

xt_world_layer info(xt_surface* s, uint32_t index) {
    xt_world_layer l;
    std::memset(&l, 0, sizeof l);
    l.struct_size = sizeof l;
    REQUIRE_MESSAGE(xt_surface_world_layer_info(s, index, &l) == XT_STATUS_OK, xt_capi_last_error());
    return l;
}

Pixels layerPixels(xt_surface* s, uint32_t index) {
    const xt_world_layer l = info(s, index);
    Pixels px(static_cast<size_t>(l.tex_width) * l.tex_height * 4u);
    REQUIRE_MESSAGE(xt_surface_world_layer_readback(s, index, px.data(),
                                                    static_cast<int>(l.tex_width * 4u)) == XT_STATUS_OK,
                    xt_capi_last_error());
    return px;
}

Pixels targetPixels(xt_gpu_target* t, int width, int height) {
    Pixels px(static_cast<size_t>(width) * static_cast<size_t>(height) * 4u);
    REQUIRE_MESSAGE(xt_gpu_target_readback(t, px.data(), width * 4) == XT_STATUS_OK, xt_capi_last_error());
    return px;
}

/// Straight alpha (xtframe-view's PNGs) back to premultiplied, as the view's unpremultiply inverted.
void premultiply(Pixels& px) {
    for (size_t i = 0; i + 3 < px.size(); i += 4) {
        const unsigned a = px[i + 3];
        for (size_t c = 0; c < 3; ++c) px[i + c] = static_cast<uint8_t>((px[i + c] * a + 127u) / 255u);
    }
}

/// The largest channel difference and how many pixels differ by more than `tolerance`.
struct Diff {
    int maxDiff = 0;
    size_t over = 0;
    size_t differing = 0;
};
Diff compare(const Pixels& a, const Pixels& b, int tolerance) {
    Diff d;
    REQUIRE(a.size() == b.size());
    for (size_t i = 0; i + 3 < a.size(); i += 4) {
        int m = 0;
        for (size_t c = 0; c < 4; ++c) m = std::max(m, std::abs(int(a[i + c]) - int(b[i + c])));
        d.maxDiff = std::max(d.maxDiff, m);
        d.over += m > tolerance ? 1u : 0u;
        d.differing += m > 0 ? 1u : 0u;
    }
    return d;
}

/// Runs `xtframe-view --layers` on the world-layers demo (headless wgpu, 1x, transparent, the given camera)
/// into `out`; false (the case is skipped) in a build without the tool.
bool runViewLayers(const std::filesystem::path& out) {
#ifdef XT_CAPI_TEST_VIEW_EXE
    std::filesystem::remove_all(out);
    char camera[128];
    std::snprintf(camera, sizeof camera, " --layers-yaw %g --layers-pitch %g --layers-depth %g", kYaw, kPitch,
                  kDepth);
    const std::string log = (out.parent_path() / (out.filename().string() + ".log")).string();
    const std::string command =
        std::string("\"\"") + XT_CAPI_TEST_VIEW_EXE + "\" \"" + XT_CAPI_TEST_WORLD_LAYERS_BUNDLE +
        "\" --ui stage --width " + std::to_string(kDemoW) + " --height " + std::to_string(kDemoH) +
        " --frames " + std::to_string(kDemoFrames) + " --layers \"" + out.string() +
        "\" --transparent --msaa 1 --renderer wgpu" + camera + " > \"" + log + "\" 2>&1\"";
    const int code = std::system(command.c_str());
    if (code != 0) {
        std::ifstream in(log);
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        FAIL("xtframe-view --layers failed (" << code << "): " << text);
    }
    return true;
#else
    (void)out;
    std::printf("SKIP: built without xtframe-view (no --layers reference)\n");
    return false;
#endif
}

nlohmann::json readJson(const std::filesystem::path& file) {
    std::ifstream in(file);
    REQUIRE_MESSAGE(in.good(), file.string());
    return nlohmann::json::parse(in);
}

Pixels readPremultipliedPng(const std::filesystem::path& file, int& width, int& height) {
    Pixels px;
    REQUIRE_MESSAGE(xt::sw::readPng(file.string(), px, width, height), file.string());
    premultiply(px);
    return px;
}

/// The index of the layer with `id`, or -1.
int findLayer(xt_surface* s, uint32_t id) {
    const uint32_t n = xt_surface_world_layer_count(s);
    for (uint32_t i = 0; i < n; ++i) {
        if (info(s, i).id == id) return static_cast<int>(i);
    }
    return -1;
}

/// (x, y, z, 1) through a row-major 4x4.
void transform(const float* m, float x, float y, float z, float out[4]) {
    for (int r = 0; r < 4; ++r) out[r] = m[r * 4] * x + m[r * 4 + 1] * y + m[r * 4 + 2] * z + m[r * 4 + 3];
}

xt_input_event pointer(int kind) {
    xt_input_event e;
    std::memset(&e, 0, sizeof e);
    e.struct_size = sizeof e;
    e.kind = kind;
    return e;
}

uint64_t surfaceBytes(xt_gpu_target* t) {
    xt_gpu_memory_stats m{};
    m.struct_size = sizeof m;
    REQUIRE(xt_gpu_target_memory_stats(t, &m) == XT_STATUS_OK);
    return m.surface_bytes;
}

}  // namespace

TEST_CASE(
    "capi world layers 1 (T-709): the demo's layers, their pixels and the orbit view equal xtframe-view") {
    Fixture f;
    xt_gpu_target* t = f.target(kDemoW, kDemoH);
    if (t == nullptr) return;
    const std::filesystem::path out = xt::test::tempDir() / "capi-world-layers-view";
    if (!runViewLayers(out)) return;
    const nlohmann::json ref = readJson(out / "layers.json");

    xt_surface* s = f.surface(XT_CAPI_TEST_WORLD_LAYERS_BUNDLE, "stage", kDemoW, kDemoH);
    REQUIRE(xt_surface_set_world_layers(s, 1) == XT_STATUS_OK);
    float m[16];
    REQUIRE(xt_world_layer_orbit_matrix(m, kDemoW, kDemoH, kDemoW, kDemoH, kYaw, kPitch, 0.0f) ==
            XT_STATUS_OK);
    uint32_t ringsId = 0;
    for (const auto& l : ref["layers"]) {
        if (l["element"] == "Rings") ringsId = l["id"].get<uint32_t>();
    }
    REQUIRE(ringsId != 0);
    for (int frame = 0; frame < kDemoFrames; ++frame) {
        layerFrame(s, t, frame == 0 ? 0.0f : 1.0f / 60.0f, m);
        if (frame >= 1) {
            // 33 §33.12: from the second frame on only the turning #Rings is redrawn; #Tags moves and #Fore
            // fades without a redraw.
            for (uint32_t i = 0; i < xt_surface_world_layer_count(s); ++i) {
                const xt_world_layer l = info(s, i);
                CAPTURE(frame);
                CAPTURE(l.id);
                CHECK(l.changed == (l.id == ringsId ? 1 : 0));
            }
        }
    }

    // The list: what the C++ path (WorldLayerBuilder in xtframe-view) gave for the same frame.
    const auto& layers = ref["layers"];
    REQUIRE(xt_surface_world_layer_count(s) == layers.size());
    uint64_t texBytes = 0;
    int worstLayer = 0;
    size_t worstOver = 0;
    for (uint32_t i = 0; i < layers.size(); ++i) {
        const auto& r = layers[i];
        const xt_world_layer l = info(s, i);
        CAPTURE(i);
        CHECK(l.id == r["id"].get<uint32_t>());
        CHECK(l.order == r["order"].get<uint32_t>());
        CHECK(l.tex_width == r["texture"][0].get<uint32_t>());
        CHECK(l.tex_height == r["texture"][1].get<uint32_t>());
        for (size_t k = 0; k < 16; ++k)
            CHECK(l.world3[k] == doctest::Approx(r["world3"][k].get<float>()).epsilon(1e-5));
        for (size_t k = 0; k < 4; ++k)
            CHECK(l.rect[k] == doctest::Approx(r["rect"][k].get<float>()).epsilon(1e-5));
        CHECK(l.opacity == doctest::Approx(r["opacity"].get<float>()).epsilon(1e-5));
        CHECK((l.backface_hidden != 0) == r["backfaceHidden"].get<bool>());
        if (i > 0) CHECK(l.order > info(s, i - 1).order);
        texBytes += static_cast<uint64_t>(l.tex_width) * l.tex_height * 4u;
        // The pixels: the view's PNG (straight alpha) premultiplied again; T-701's layer tolerance is 2.
        int w = 0, h = 0;
        const Pixels png = readPremultipliedPng(out / r["file"].get<std::string>(), w, h);
        REQUIRE(w == static_cast<int>(l.tex_width));
        REQUIRE(h == static_cast<int>(l.tex_height));
        const Diff d = compare(layerPixels(s, i), png, 2);
        std::printf("world layer %u (id %u, %ux%u): max channel diff %d, %zu px > 2, %zu px differ\n", i,
                    l.id, l.tex_width, l.tex_height, d.maxDiff, d.over, d.differing);
        CHECK(d.over == 0u);
        worstLayer = std::max(worstLayer, d.maxDiff);
        worstOver += d.over;
    }
    std::printf("world layers vs xtframe-view --layers: worst channel diff %d, %zu px over 2\n", worstLayer,
                worstOver);
    // 50 §50.af item 2: the layer textures are the surface's memory (it never drew flat: nothing else).
    CHECK(surfaceBytes(t) == texBytes);

    // The orbit view: draw_layers_gpu with xt_world_layer_orbit_matrix against the view's preview.png (the
    // same camera through LayerOrbit::boxToSurface, drawn into a transparent frame).
    int pw = 0, ph = 0;
    const Pixels preview = readPremultipliedPng(out / "preview.png", pw, ph);
    REQUIRE(pw == kDemoW);
    REQUIRE(ph == kDemoH);
    const Diff d = compare(targetPixels(t, kDemoW, kDemoH), preview, 2);
    std::printf("draw_layers_gpu vs xtframe-view preview: max channel diff %d, %zu px > 2, %zu px differ\n",
                d.maxDiff, d.over, d.differing);
    CHECK(d.over == 0u);
}

TEST_CASE(
    "capi world layers 2 (T-709): static frames redraw nothing, one change one layer, flat restoration") {
    Fixture f;
    xt_gpu_target* t = f.target(160, 96);
    if (t == nullptr) return;
    xt_surface* s = f.surface(XT_CAPI_TEST_LAYERS_BUNDLE, "cards", 160, 96);
    REQUIRE(xt_surface_set_world_layers(s, 1) == XT_STATUS_OK);
    xt_surface_set_perf_enabled(s, 1);
    layerFrame(s, t, 0.0f);
    REQUIRE(xt_surface_world_layer_count(s) == 3u);  // the base layer, #left, #right
    for (uint32_t i = 0; i < 3; ++i) CHECK(info(s, i).changed == 1);
    const uint32_t rightId = info(s, 2).id;
    CHECK(info(s, 0).id == 0u);
    // A UI standing still: no layer redrawn, no frame built (so nothing submitted).
    for (int frame = 1; frame < 3; ++frame) {
        layerFrame(s, t, 1.0f / 60.0f);
        for (uint32_t i = 0; i < 3; ++i) CHECK(info(s, i).changed == 0);
        xt_frame_stats stats{};
        REQUIRE(xt_surface_frame_stats(s, &stats) == XT_STATUS_OK);
        CHECK(stats.executed_draws == 0u);
    }
    // One element inside #right changes: only that layer.
    const uint32_t dot = xt_surface_port_lookup(s, "dot");
    REQUIRE(dot != XT_INVALID_PORT);
    REQUIRE(xt_surface_port_set_bool(s, dot, 0) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f);
    for (uint32_t i = 0; i < 3; ++i) CHECK(info(s, i).changed == (info(s, i).id == rightId ? 1 : 0));
    REQUIRE(xt_surface_port_set_bool(s, dot, 1) == XT_STATUS_OK);

    // Flat restoration (T-701): depth 0 and the identity put the coplanar layers back on top of one another;
    // the result is the ordinary GPU picture of the same UI (the wgpu tolerance of T-701 is 2).
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    layerFrame(s, t, 1.0f / 60.0f, identity, 0.0f);
    const Pixels layered = targetPixels(t, 160, 96);
    xt_gpu_target* flatTarget = f.target(160, 96);
    REQUIRE(flatTarget != nullptr);
    xt_surface* flat = f.surface(XT_CAPI_TEST_LAYERS_BUNDLE, "cards", 160, 96);
    REQUIRE(xt_surface_update(flat, 0.0f) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_begin_frame(flatTarget, nullptr) == XT_STATUS_OK);
    REQUIRE(xt_surface_render_gpu(flat, flatTarget, nullptr) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_end_frame(flatTarget) == XT_STATUS_OK);
    const Diff d = compare(layered, targetPixels(flatTarget, 160, 96), 2);
    std::printf("depth 0 + identity vs render_gpu: max channel diff %d, %zu px > 2, %zu px differ\n",
                d.maxDiff, d.over, d.differing);
    CHECK(d.over == 0u);
    // A dest offset moves the picture (and clips to dest).
    REQUIRE(xt_surface_update(s, 0.0f) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_begin_frame(t, nullptr) == XT_STATUS_OK);
    REQUIRE(xt_surface_render_layers_gpu(s, t) == XT_STATUS_OK);
    const xt_rect half{80, 0, 80, 96};
    REQUIRE(xt_surface_draw_layers_gpu(s, t, &half, identity, 0.0f) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_end_frame(t) == XT_STATUS_OK);
    const Pixels moved = targetPixels(t, 160, 96);
    for (int y = 0; y < 96; y += 7) {
        CHECK(moved[(static_cast<size_t>(y) * 160 + 10) * 4 + 3] == 0);  // left of dest: nothing
        CHECK(std::memcmp(&moved[(static_cast<size_t>(y) * 160 + 85) * 4],
                          &layered[(static_cast<size_t>(y) * 160 + 5) * 4], 4) == 0);
    }
}

TEST_CASE(
    "capi world layers 3 (T-709): a ray through the orbit camera reaches ENGAGE; the ghost spot does not") {
    Fixture f;
    xt_gpu_target* t = f.target(kDemoW, kDemoH);
    if (t == nullptr) return;
    xt_surface* s = f.surface(XT_CAPI_TEST_WORLD_LAYERS_BUNDLE, "stage", kDemoW, kDemoH);
    REQUIRE(xt_surface_set_world_layers(s, 1) == XT_STATUS_OK);
    float m[16];
    REQUIRE(xt_world_layer_orbit_matrix(m, kDemoW, kDemoH, kDemoW, kDemoH, kYaw, kPitch, 0.0f) ==
            XT_STATUS_OK);
    layerFrame(s, t, 0.0f, m);
    layerFrame(s, t, 1.0f / 60.0f, m);
    // The panel layer: the one whose rect holds the 440 x 380 panel (stage.layout.xts).
    int panel = -1;
    for (uint32_t i = 0; i < xt_surface_world_layer_count(s); ++i) {
        const xt_world_layer l = info(s, i);
        if (l.id != 0 && l.rect[0] <= 0.0f && l.rect[1] <= 0.0f && l.rect[0] + l.rect[2] >= 440.0f &&
            l.rect[1] + l.rect[3] >= 380.0f && l.rect[2] < 600.0f) {
            panel = static_cast<int>(i);
        }
    }
    REQUIRE(panel >= 0);
    const xt_world_layer pl = info(s, static_cast<uint32_t>(panel));
    const uint32_t panelId = pl.id;
    // ENGAGE's centre on the panel plane (stage.layout.xts: #Engage at 32, 292, 376 x 56 in #Panel).
    const float ex = 32.0f + 188.0f, ey = 292.0f + 28.0f;
    const float density = static_cast<float>(pl.tex_width) / pl.rect[2];
    const int tx = static_cast<int>((ex - pl.rect[0]) * density),
              ty = static_cast<int>((ey - pl.rect[1]) * density);
    auto engageTexel = [&]() {
        const int index = findLayer(s, panelId);
        REQUIRE(index >= 0);
        const Pixels px = layerPixels(s, static_cast<uint32_t>(index));
        const size_t at = (static_cast<size_t>(ty) * info(s, static_cast<uint32_t>(index)).tex_width +
                           static_cast<size_t>(tx)) *
                          4u;
        return std::vector<uint8_t>(px.begin() + static_cast<std::ptrdiff_t>(at),
                                    px.begin() + static_cast<std::ptrdiff_t>(at) + 4);
    };
    auto panelChanged = [&]() { return info(s, static_cast<uint32_t>(findLayer(s, panelId))).changed != 0; };
    const std::vector<uint8_t> idle = engageTexel();
    CHECK(idle[0] == 0x7f);  // #7fe3ff
    CHECK(idle[2] == 0xff);
    auto rayAt = [&](float rootX, float rootY, float rootZ, float origin[3], float direction[3]) {
        float p[4];
        transform(m, rootX, rootY, rootZ * kDepth, p);
        REQUIRE(p[3] > 0.0f);
        REQUIRE(xt_world_layer_unproject(m, kDepth, p[0] / p[3], p[1] / p[3], origin, direction) ==
                XT_STATUS_OK);
    };

    // T-703: the ghost spot -- ENGAGE where it would be without the layer's transform (root box 660 + ex,
    // 150 + ey, z 0) -- must not hit it. The premise, checked first: that ray crosses the panel's real plane
    // outside ENGAGE.
    float o[3], d[3];
    rayAt(660.0f + ex, 150.0f + ey, 0.0f, o, d);
    {
        // o + t d = world3 (u, v, 0): u c0 + v c1 - t d = o - c3, solved by Cramer's rule.
        const float* w = pl.world3;
        const float a[3][3] = {{w[0], w[1], -d[0]}, {w[4], w[5], -d[1]}, {w[8], w[9], -d[2]}};
        const float b[3] = {o[0] - w[3], o[1] - w[7], o[2] - w[11]};
        auto det3 = [](const float mm[3][3]) {
            return mm[0][0] * (mm[1][1] * mm[2][2] - mm[1][2] * mm[2][1]) -
                   mm[0][1] * (mm[1][0] * mm[2][2] - mm[1][2] * mm[2][0]) +
                   mm[0][2] * (mm[1][0] * mm[2][1] - mm[1][1] * mm[2][0]);
        };
        const float det = det3(a);
        REQUIRE(std::fabs(det) > 1e-6f);
        float au[3][3], av[3][3];
        std::memcpy(au, a, sizeof a);
        std::memcpy(av, a, sizeof a);
        for (int r = 0; r < 3; ++r) {
            au[r][0] = b[r];
            av[r][1] = b[r];
        }
        const float u = det3(au) / det, v = det3(av) / det;
        CAPTURE(u);
        CAPTURE(v);
        REQUIRE_FALSE((u >= 32.0f && u <= 408.0f && v >= 292.0f && v <= 348.0f));
    }
    xt_input_event move = pointer(XT_INPUT_POINTER_MOVE);
    REQUIRE(xt_surface_push_input_ray(s, &move, o, d) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f, m);
    CHECK_FALSE(panelChanged());

    // ENGAGE itself: its centre on the panel plane through world3, then the camera, then back to a ray.
    float q[4];
    transform(pl.world3, ex, ey, 0.0f, q);
    rayAt(q[0], q[1], q[2], o, d);
    REQUIRE(xt_surface_push_input_ray(s, &move, o, d) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f, m);
    CHECK(panelChanged());
    const std::vector<uint8_t> hover = engageTexel();  // #Engage:hover = #ffffff
    CHECK(hover[0] == 0xff);
    CHECK(hover[1] == 0xff);
    CHECK(hover[2] == 0xff);
    xt_input_event down = pointer(XT_INPUT_POINTER_DOWN);
    REQUIRE(xt_surface_push_input_ray(s, &down, o, d) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f, m);
    const std::vector<uint8_t> active = engageTexel();  // #Engage:active = #ffd166
    CHECK(active[0] == 0xff);
    CHECK(active[1] == 0xd1);
    CHECK(active[2] == 0x66);
    xt_input_event up = pointer(XT_INPUT_POINTER_UP);
    CHECK(xt_surface_push_input_ray(s, &up, o, d) == XT_STATUS_OK);
}

TEST_CASE("capi world layers 4 (T-709): preconditions, NULL, ranges and releases") {
    Fixture f;
    xt_gpu_target* t = f.target(160, 96);
    if (t == nullptr) return;
    xt_surface* s = f.surface(XT_CAPI_TEST_LAYERS_BUNDLE, "cards", 160, 96);
    const float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    const float o[3] = {20, 20, 100}, d[3] = {0, 0, -1};
    xt_input_event move = pointer(XT_INPUT_POINTER_MOVE);
    // NULL.
    CHECK(xt_surface_set_world_layers(nullptr, 1) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_set_world_layer_density(nullptr, 1) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_render_layers_gpu(nullptr, t) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_render_layers_gpu(s, nullptr) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_world_layer_count(nullptr) == 0u);
    CHECK(xt_surface_world_layer_info(nullptr, 0, nullptr) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_world_layer_info(s, 0, nullptr) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_world_layer_readback(nullptr, 0, nullptr, 0) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_draw_layers_gpu(nullptr, t, nullptr, identity, 0) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_draw_layers_gpu(s, t, nullptr, nullptr, 0) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_push_input_ray(nullptr, &move, o, d) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_push_input_ray(s, nullptr, o, d) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_push_input_ray(s, &move, nullptr, d) == XT_STATUS_INVALID_ARGUMENT);
    // World layers off.
    REQUIRE(xt_gpu_target_begin_frame(t, nullptr) == XT_STATUS_OK);
    CHECK(xt_surface_render_layers_gpu(s, t) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(contains(xt_capi_last_error(), "world layers are off"));
    CHECK(xt_surface_draw_layers_gpu(s, t, nullptr, identity, 0) == XT_STATUS_INVALID_ARGUMENT);
    REQUIRE(xt_gpu_target_end_frame(t) == XT_STATUS_OK);
    CHECK(xt_surface_push_input_ray(s, &move, o, d) == XT_STATUS_INVALID_ARGUMENT);
    // On: no flat picture; density checked; out of the frame refused.
    REQUIRE(xt_surface_set_world_layers(s, 1) == XT_STATUS_OK);
    CHECK(xt_surface_set_world_layer_density(s, 0.0f) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_set_world_layer_density(s, std::nanf("")) == XT_STATUS_INVALID_ARGUMENT);
    std::vector<uint8_t> sw(160u * 96u * 4u);
    CHECK(xt_surface_render(s, sw.data(), 160 * 4, nullptr) == XT_STATUS_INVALID_ARGUMENT);
    REQUIRE(xt_surface_update(s, 0.0f) == XT_STATUS_OK);
    CHECK(xt_surface_render_layers_gpu(s, t) == XT_STATUS_INVALID_ARGUMENT);  // outside a frame
    CHECK(contains(xt_capi_last_error(), "outside"));
    xt_input_event text = pointer(XT_INPUT_KEY_DOWN);
    CHECK(xt_surface_push_input_ray(s, &text, o, d) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_push_input_ray(s, &move, o, d) == XT_STATUS_OK);
    REQUIRE(xt_gpu_target_begin_frame(t, nullptr) == XT_STATUS_OK);
    CHECK(xt_surface_render_gpu(s, t, nullptr) == XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_draw_layers_gpu(s, t, nullptr, identity, 0) ==
          XT_STATUS_INVALID_ARGUMENT);  // before render_layers
    REQUIRE(xt_surface_render_layers_gpu(s, t) == XT_STATUS_OK);
    CHECK(xt_surface_draw_layers_gpu(s, t, nullptr, identity, -1.0f) == XT_STATUS_INVALID_ARGUMENT);
    const xt_rect empty{0, 0, 0, 10};
    CHECK(xt_surface_draw_layers_gpu(s, t, &empty, identity, 0) == XT_STATUS_INVALID_ARGUMENT);
    const uint32_t n = xt_surface_world_layer_count(s);
    REQUIRE(n == 3u);
    const xt_world_layer first = info(s, 0);
    std::vector<uint8_t> px(static_cast<size_t>(first.tex_width) * first.tex_height * 4u);
    CHECK(xt_surface_world_layer_readback(s, 0, px.data(), static_cast<int>(first.tex_width * 4)) ==
          XT_STATUS_INVALID_ARGUMENT);  // inside the frame that built it
    REQUIRE(xt_gpu_target_end_frame(t) == XT_STATUS_OK);
    CHECK(xt_surface_world_layer_readback(s, 0, px.data(), static_cast<int>(first.tex_width * 4)) ==
          XT_STATUS_OK);
    CHECK(xt_surface_world_layer_readback(s, 0, px.data(), static_cast<int>(first.tex_width * 4) - 1) ==
          XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_world_layer_readback(s, 0, nullptr, static_cast<int>(first.tex_width * 4)) ==
          XT_STATUS_INVALID_ARGUMENT);
    CHECK(xt_surface_world_layer_readback(s, n, px.data(), 1 << 20) == XT_STATUS_OUT_OF_RANGE);
    xt_world_layer l{};
    l.struct_size = sizeof l;
    CHECK(xt_surface_world_layer_info(s, n, &l) == XT_STATUS_OUT_OF_RANGE);
    // A shorter struct gets its prefix only.
    xt_world_layer shortOne;
    std::memset(&shortOne, 0xAB, sizeof shortOne);
    shortOne.struct_size = static_cast<uint32_t>(offsetof(xt_world_layer, tex_width));
    REQUIRE(xt_surface_world_layer_info(s, 2, &shortOne) == XT_STATUS_OK);
    CHECK(shortOne.id == info(s, 2).id);
    CHECK(shortOne.tex_width == 0xABABABABu);
    // A second target is refused while attached.
    xt_gpu_target* other = f.target(16, 16);
    REQUIRE(other != nullptr);
    REQUIRE(xt_gpu_target_begin_frame(other, nullptr) == XT_STATUS_OK);
    CHECK(xt_surface_render_layers_gpu(s, other) == XT_STATUS_INVALID_ARGUMENT);
    REQUIRE(xt_gpu_target_end_frame(other) == XT_STATUS_OK);
    // Releases: the layer textures count in surface_bytes until trim / hide / off / detach.
    CHECK(surfaceBytes(t) > 0u);
    REQUIRE(xt_surface_trim(s) == XT_STATUS_OK);
    CHECK(surfaceBytes(t) == 0u);
    CHECK(xt_surface_world_layer_count(s) == 0u);
    layerFrame(s, t, 0.0f);
    CHECK(xt_surface_world_layer_count(s) == 3u);
    CHECK(surfaceBytes(t) > 0u);
    REQUIRE(xt_surface_set_visible(s, 0) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f);  // the update that hides it releases (trim-on-hide)
    CHECK(xt_surface_world_layer_count(s) == 0u);
    CHECK(surfaceBytes(t) == 0u);
    REQUIRE(xt_surface_set_visible(s, 1) == XT_STATUS_OK);
    layerFrame(s, t, 1.0f / 60.0f);
    CHECK(xt_surface_world_layer_count(s) == 3u);
    for (uint32_t i = 0; i < 3; ++i) CHECK(info(s, i).changed == 1);
    REQUIRE(xt_surface_set_world_layers(s, 0) == XT_STATUS_OK);
    CHECK(xt_surface_world_layer_count(s) == 0u);
    CHECK(surfaceBytes(t) == 0u);
    REQUIRE(xt_surface_set_world_layers(s, 1) == XT_STATUS_OK);
    layerFrame(s, t, 0.0f);
    CHECK(surfaceBytes(t) > 0u);
    REQUIRE(xt_surface_detach_gpu(s) == XT_STATUS_OK);
    CHECK(surfaceBytes(t) == 0u);
    CHECK(xt_surface_world_layer_count(s) == 0u);
    // Density 2: every texture twice the size.
    REQUIRE(xt_surface_set_world_layer_density(s, 2.0f) == XT_STATUS_OK);
    layerFrame(s, t, 0.0f);
    const xt_world_layer base = info(s, 0);
    CHECK(base.tex_width == static_cast<uint32_t>(std::ceil(base.rect[2] * 2.0f)));
    CHECK(base.tex_height == static_cast<uint32_t>(std::ceil(base.rect[3] * 2.0f)));
}

#endif  // XT_CAPI_TEST_HAS_WGPU
