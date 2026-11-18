// T-728 (Docs/design/10-backends.md §10.18 A item 1, Docs/tasks/T-728.md "測試"): the AlphaExtract
// pass folded into the next Downsample / GaussScalar pass (GaussParams / DownsampleParams `channel`)
// gives the same bytes as the unfolded chain. Every case is drawn twice by xt_wgpu -- once by a
// device that reports supportsAlphaChannelReads() (one pass less per folded effect) and once by one
// that does not (the frames of before) -- and the two read-backs must be identical byte for byte.
// The source holds every alpha value 0..255 and antialiased polygon edges, so the blur taps filter
// arbitrary alpha neighbourhoods; the blurs cover the full-resolution and the downsampled paths.
// T-735: inset shadows and inner glows (`1 - .a`) are not folded -- same passes, same bytes.

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
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

constexpr int kW = 256, kH = 192;

/// xt_wgpu as a device that does not report channel reads (a host's device that has not caught up).
class NoChannelDevice : public wgpu::WgpuDevice {
public:
    bool supportsAlphaChannelReads() const override { return false; }
};

/// The effect group's content: 256 one-pixel columns carrying every alpha value, a translucent
/// 7-gon and scattered translucent triangles (antialiased edges of arbitrary coverage).
CommandList foldList(std::initializer_list<EffectInstance> effects) {
    CommandList cl = makeList(kW, kH);
    addFillRect(cl, 0, 0, kW, kH, Color{0.2f, 0.35f, 0.5f, 1});  // below the group
    addPushEffect(cl, effects, Rect{0, 0, kW, kH}, Rect{40, 36, 176, 120});
    for (int i = 0; i < 128; ++i) {
        const float x = 64.0f + static_cast<float>(i);
        addFillRect(cl, x, 44, 1, 22,
                    Color{static_cast<float>(i) / 127.0f, 0.4f, 0.9f, static_cast<float>(i) / 255.0f});
        addFillRect(cl, x, 70, 1, 22,
                    Color{0.9f, static_cast<float>(i) / 127.0f, 0.3f, static_cast<float>(128 + i) / 255.0f});
    }
    PathRef gon = emptyPath(cl);
    appendNgon(cl, gon, Vec2{96.3f, 124.7f}, 23.6f, 7);
    addFill(cl, gon, Color{1, 1, 1, 0.73f});
    uint32_t seed = 12345u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / 16777216.0f;
    };
    for (int t = 0; t < 24; ++t) {
        const float cx = 130.0f + next() * 80.0f;
        const float cy = 100.0f + next() * 50.0f;
        addFill(cl,
                addPolygon(cl, {Vec2{cx + next() * 14.0f, cy - next() * 13.0f},
                                Vec2{cx - next() * 15.0f, cy + next() * 3.0f},
                                Vec2{cx + next() * 6.0f, cy + next() * 16.0f}}),
                Color{next(), next(), next(), 0.15f + next() * 0.85f});
    }
    addPopEffect(cl);
    return cl;
}

EffectInstance dropShadow(float blur, float spread, bool inset) {
    EffectInstance e = effect(EffectKind::DropShadow, {4, 5, blur, spread, 1, inset ? 1.0f : 0.0f},
                              Color{0.05f, 0, 0.1f, 0.85f});
    e.enabled = 1;
    return e;
}

EffectInstance glow(float radius, bool inner) {
    EffectInstance e =
        effect(EffectKind::Glow, {radius, 1.4f, inner ? 1.0f : 0.0f}, Color{1, 0.85f, 0.2f, 1});
    e.enabled = 1;
    return e;
}

struct Drawn {
    std::vector<uint8_t> px;
    size_t passes = 0;
};

Drawn draw(const CommandList& cl, bool channels) {
    Session s(kW, kH);  // its target and readback
    std::unique_ptr<wgpu::WgpuDevice> device =
        channels ? std::make_unique<wgpu::WgpuDevice>() : std::make_unique<NoChannelDevice>();
    std::string err;
    REQUIRE_MESSAGE(
        device->init(sharedInit().device, sharedInit().queue, WGPUTextureFormat_RGBA8Unorm, 1, err), err);
    device->resize(kW, kH);
    gpu::FrameBuilder builder;
    builder.init(*device);
    builder.resize(kW, kH);
    builder.setEdgeFringe(true);
    gpu::GpuFrame frame;
    builder.build(cl, nullptr, nullptr, nullptr, frame);
    Drawn out;
    out.passes = frame.passes.size();
    device->setSurfaceClear(true, WGPUColor{0, 0, 0, 1});
    device->setSurface(s.target.view);
    device->submit(frame);
    device->setSurface(nullptr);
    CHECK_MESSAGE(std::string(device->lastError()).empty(), device->lastError());
    out.px = s.read();
    builder.shutdown();
    device->shutdown();
    return out;
}

void checkSame(const char* what, std::initializer_list<EffectInstance> effects, size_t folded) {
    CAPTURE(what);
    const CommandList cl = foldList(effects);
    const Drawn before = draw(cl, false);
    const Drawn after = draw(cl, true);
    CHECK(after.passes + folded == before.passes);
    REQUIRE(after.px.size() == before.px.size());
    size_t differ = 0;
    for (size_t i = 0; i < after.px.size(); i += 4) {
        differ += (after.px[i] != before.px[i] || after.px[i + 1] != before.px[i + 1] ||
                   after.px[i + 2] != before.px[i + 2] || after.px[i + 3] != before.px[i + 3])
                      ? 1u
                      : 0u;
    }
    std::printf("wgpu channel fold %s: %zu -> %zu passes, %zu px differ\n", what, before.passes, after.passes,
                differ);
    CHECK(differ == 0);
}

}  // namespace

TEST_CASE("wgpu channel fold: outer drop shadows are byte-identical with one pass less (T-728)") {
    checkSame("outer blur 6", {dropShadow(6, 0, false)}, 1);
    checkSame("outer blur 24 (downsampled)", {dropShadow(24, 0, false)}, 1);
    checkSame("outer blur 64 (eighth)", {dropShadow(64, 0, false)}, 1);
}

TEST_CASE("wgpu channel fold: inset drop shadows keep their passes and are byte-identical (T-735)") {
    checkSame("inset blur 6", {dropShadow(6, 0, true)}, 0);
    checkSame("inset blur 24 (downsampled)", {dropShadow(24, 0, true)}, 0);
    checkSame("inset blur 64 (eighth)", {dropShadow(64, 0, true)}, 0);
}

TEST_CASE("wgpu channel fold: glows fold one pass, inner glows none; both byte-identical (T-728, T-735)") {
    checkSame("glow 8", {glow(8, false)}, 1);
    checkSame("glow 30 (downsampled)", {glow(30, false)}, 1);
    checkSame("inner glow 8", {glow(8, true)}, 0);
    checkSame("inner glow 30 (downsampled)", {glow(30, true)}, 0);
}

TEST_CASE("wgpu channel fold: a stack folds its outer chains; a spread folds none (T-728, T-735)") {
    checkSame("outer + inset + glow", {dropShadow(10, 0, false), dropShadow(12, 0, true), glow(6, false)}, 2);
    checkSame("spread 2 outer, spread 3 inset", {dropShadow(6, 2, false), dropShadow(6, 3, true)}, 0);
}
