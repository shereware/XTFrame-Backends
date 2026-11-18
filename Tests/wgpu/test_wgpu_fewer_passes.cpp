// T-728 (Docs/design/10-backends.md §10.18 A, Docs/tasks/T-728.md "測試"): passes xt_wgpu no longer
// encodes. Item 2: a copy pass from one transient into another (copyTextureToTexture) that draws and
// clears nothing opens no render pass after the copy. Item 3: a glass panel (BackdropBlur, then an
// outer and an inset drop shadow) on an empty layer builds no pyramid; its pixels are byte-identical
// to those of the same list with a fully transparent fill under the panel first, which keeps the
// pyramid (an all-zero one) -- and a panel over a real drawing keeps it too.

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/wgpu/wgpu_device.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 160, kH = 120;

/// A glass panel (BackdropBlur) inside a layer: its pyramid's base is copied from the layer's
/// transient, which already holds a drawing under the panel.
CommandList layeredGlassList() {
    CommandList cl = makeList(kW, kH);
    addFillRect(cl, 0, 0, kW, kH, Color{0.15f, 0.2f, 0.3f, 1});
    addPushLayer(cl, 0.9f, BlendMode::Normal, Rect{10, 10, 140, 100});
    addFillRect(cl, 20, 20, 60, 70, Color{0.9f, 0.4f, 0.1f, 1});
    addFillRect(cl, 70, 40, 60, 50, Color{0.1f, 0.8f, 0.4f, 0.8f});
    const Rect glass{40, 30, 70, 50};
    addPushEffect(cl, {effect(EffectKind::BackdropBlur, {8})}, glass, glass);
    addFillRect(cl, glass.x, glass.y, glass.w, glass.h, Color{1, 1, 1, 0.25f});
    addPopEffect(cl);
    addPopLayer(cl);
    return cl;
}

EffectInstance glassEffect(EffectKind kind, std::initializer_list<float> params,
                           Color color = Color{0, 0, 0, 1}) {
    EffectInstance e = effect(kind, params, color);
    e.enabled = 1;
    return e;
}

/// A layer over a striped surface holding a glass panel; `under` is drawn into the layer before the
/// panel (alpha 0: draws nothing, but the region read is no longer untouched).
CommandList glassOnLayerList(bool under, float underAlpha) {
    CommandList cl = makeList(kW, kH);
    for (int i = 0; i < 8; ++i) {
        addFillRect(cl, static_cast<float>(i) * 20.0f, 0, 20, kH,
                    Color{0.1f * static_cast<float>(i), 0.3f, 0.9f - 0.1f * static_cast<float>(i), 1});
    }
    addPushLayer(cl, 0.9f, BlendMode::Normal, Rect{0, 0, kW, kH});
    if (under) {
        addFillRect(cl, 20, 22, 120, 76, Color{0.9f, 0.8f, 0.1f, underAlpha});
    }
    const Rect panel{36, 34, 88, 52};
    addPushEffect(cl,
                  {glassEffect(EffectKind::BackdropBlur, {10, 0.4f, 0.1f}),
                   glassEffect(EffectKind::DropShadow, {0, 5, 10, 0, 1, 0}, Color{0, 0, 0, 0.55f}),
                   glassEffect(EffectKind::DropShadow, {2, 3, 6, 0, 1, 1}, Color{1, 1, 1, 0.6f})},
                  panel, panel);
    addFillRect(cl, panel.x, panel.y, panel.w, panel.h, Color{1, 1, 1, 0.22f});
    PathRef gon = emptyPath(cl);
    appendNgon(cl, gon, Vec2{80.4f, 60.3f}, 17.7f, 9);
    addFill(cl, gon, Color{0.2f, 0.9f, 0.6f, 0.65f});
    addPopEffect(cl);
    addFillRect(cl, 130, 90, 20, 20, Color{1, 0.2f, 0.2f, 1});  // after the panel
    addPopLayer(cl);
    return cl;
}

struct Built {
    std::vector<uint8_t> px;
    size_t passes = 0;
    uint32_t copies = 0;
};

Built drawFrame(const CommandList& cl) {
    Session s(kW, kH);
    wgpu::WgpuDevice device;
    std::string err;
    REQUIRE_MESSAGE(
        device.init(sharedInit().device, sharedInit().queue, WGPUTextureFormat_RGBA8Unorm, 1, err), err);
    device.resize(kW, kH);
    gpu::FrameBuilder builder;
    builder.init(device);
    builder.resize(kW, kH);
    builder.setEdgeFringe(true);
    gpu::GpuFrame frame;
    builder.build(cl, nullptr, nullptr, nullptr, frame);
    Built out;
    out.passes = frame.passes.size();
    for (const gpu::Pass& p : frame.passes) {
        out.copies += p.copy != 0 ? 1u : 0u;
    }
    device.setSurfaceClear(true, WGPUColor{0, 0, 0, 1});
    device.setSurface(s.target.view);
    device.submit(frame);
    device.setSurface(nullptr);
    CHECK_MESSAGE(std::string(device.lastError()).empty(), device.lastError());
    out.px = s.read();
    builder.shutdown();
    device.shutdown();
    return out;
}

}  // namespace

TEST_CASE(
    "wgpu fewer passes: a glass panel on an empty layer is byte-identical without its pyramid (T-728 "
    "item 3)") {
    const Built shortcut = drawFrame(glassOnLayerList(false, 0));
    const Built pyramid = drawFrame(glassOnLayerList(true, 0));  // a transparent fill first
    CHECK(shortcut.copies == 0u);
    CHECK(pyramid.copies == 1u);
    std::printf("wgpu fewer passes, item 3: %zu passes with the pyramid, %zu without\n", pyramid.passes,
                shortcut.passes);
    CHECK(shortcut.passes < pyramid.passes);
    CHECK(shortcut.px == pyramid.px);
    // Over a real drawing the pyramid stays, and the panel blurs it (SW's tolerance).
    const CommandList over = glassOnLayerList(true, 1);
    const Built drawn = drawFrame(over);
    CHECK(drawn.copies == 1u);
    CHECK(matches(drawn.px, renderSw(over, kW, kH, Color{0, 0, 0, 1}), kW, kH, "glass over a drawing"));
    CHECK(matches(shortcut.px, renderSw(glassOnLayerList(false, 0), kW, kH, Color{0, 0, 0, 1}), kW, kH,
                  "glass on an empty layer"));
}

TEST_CASE("wgpu fewer passes: no render pass after a transient copy that draws nothing (T-728 item 2)") {
    const CommandList cl = layeredGlassList();
    Session s(kW, kH);
    wgpu::WgpuDevice device;
    std::string err;
    REQUIRE_MESSAGE(
        device.init(sharedInit().device, sharedInit().queue, WGPUTextureFormat_RGBA8Unorm, 1, err), err);
    device.resize(kW, kH);
    gpu::FrameBuilder builder;
    builder.init(device);
    builder.resize(kW, kH);
    builder.setEdgeFringe(true);
    gpu::GpuFrame frame;
    builder.build(cl, nullptr, nullptr, nullptr, frame);
    uint32_t transientCopies = 0, copyOnly = 0;
    for (const gpu::Pass& p : frame.passes) {
        if (p.copy != 0 && p.copySource != gpu::kSurfaceTarget) {
            ++transientCopies;
            copyOnly += (p.cmdCount == 0 && p.clear == 0) ? 1 : 0;
        }
    }
    CHECK(transientCopies >= 1u);
    CHECK(copyOnly >= 1u);
    device.setSurfaceClear(true, WGPUColor{0, 0, 0, 1});
    device.setSurface(s.target.view);
    device.submit(frame);
    device.setSurface(nullptr);
    CHECK_MESSAGE(std::string(device.lastError()).empty(), device.lastError());
    const wgpu::WgpuDeviceStats& d = device.lastStats();
    CHECK(d.copies == transientCopies);
    CHECK(d.blits == 0u);
    std::printf("wgpu fewer passes, item 2: %zu frame passes, %u render passes, %u copy-only\n",
                frame.passes.size(), d.passes, copyOnly);
    CHECK(d.passes + copyOnly == static_cast<uint32_t>(frame.passes.size()));
    const std::vector<uint8_t> gpu = s.read();
    CHECK(matches(gpu, renderSw(cl, kW, kH, Color{0, 0, 0, 1}), kW, kH, "glass in a layer"));
    builder.shutdown();
    device.shutdown();
}
