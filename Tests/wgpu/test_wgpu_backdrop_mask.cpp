// T-729 (Docs/design/10-backends.md §10.18 "B 定案", Docs/tasks/T-729.md "測試"): xt_wgpu draws a
// stack of BackdropBlur followed only by shadow-type effects with the new semantics (the shadows from
// the element's own content, then R' over the blurred backdrop masked by the content's alpha, one
// EffectProgram::BackdropMasked pass) within 15 §15.3's tolerance of xt::sw; every other stack, and
// every stack on a device that does not report supportsBackdropMask(), is drawn byte for byte as
// before (the frames do not change); the masked composite adds no pass.

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

/// xt_wgpu as a device that does not report the backdrop mask (a host's device that has not caught up).
class NoMaskDevice : public wgpu::WgpuDevice {
public:
    bool supportsBackdropMask() const override { return false; }
};

EffectInstance on(EffectInstance e) {
    e.enabled = 1;
    return e;
}
EffectInstance blurFx() { return on(effect(EffectKind::BackdropBlur, {10, 0.4f, 0.1f})); }
EffectInstance outerFx() {
    return on(effect(EffectKind::DropShadow, {3, 7, 12, 0, 1, 0}, Color{0.05f, 0, 0.1f, 0.75f}));
}
EffectInstance insetFx() {
    return on(effect(EffectKind::DropShadow, {0, 3, 8, 0, 1, 1}, Color{0, 0, 0.2f, 0.6f}));
}
EffectInstance glowFx(bool inner) {
    return on(effect(EffectKind::Glow, {10, 1.3f, inner ? 1.0f : 0.0f}, Color{1, 0.85f, 0.2f, 1}));
}
EffectInstance adjustFx() { return on(effect(EffectKind::ColorAdjust, {0.1f, 0, -0.4f, 0, 1})); }

/// Hard-edged stripes, then a translucent panel (a white fill at alpha 0.5 and a translucent triangle).
CommandList glassList(std::initializer_list<EffectInstance> effects) {
    CommandList cl = makeList(kW, kH);
    for (int i = 0; i < 8; ++i) {
        addFillRect(cl, static_cast<float>(i) * 32.0f, 0, 32, kH,
                    Color{0.12f * static_cast<float>(i), 0.35f, 0.9f - 0.1f * static_cast<float>(i), 1});
    }
    const Rect panel{64, 48, 128, 80};
    addPushEffect(cl, effects, Rect{34, 18, 188, 140}, panel);
    addFillRect(cl, panel.x, panel.y, panel.w, panel.h, Color{1, 1, 1, 0.5f});
    addFill(cl, addPolygon(cl, {Vec2{90.3f, 60.6f}, Vec2{170.2f, 70.1f}, Vec2{120.7f, 118.4f}}),
            Color{0.2f, 0.9f, 0.6f, 0.65f});
    addPopEffect(cl);
    return cl;
}

struct Drawn {
    std::vector<uint8_t> px;
    size_t passes = 0;
    uint32_t masked = 0;  ///< BackdropMasked commands
};

Drawn draw(const CommandList& cl, bool mask) {
    Session s(kW, kH);  // its target and readback
    std::unique_ptr<wgpu::WgpuDevice> device =
        mask ? std::make_unique<wgpu::WgpuDevice>() : std::make_unique<NoMaskDevice>();
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
    for (const gpu::DrawCmd& c : frame.cmds) {
        out.masked +=
            c.shading == gpu::Shading::Effect && c.program == gpu::EffectProgram::BackdropMasked ? 1u : 0u;
    }
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

size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
        n += (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2] || a[i + 3] != b[i + 3]) ? 1u : 0u;
    }
    return n;
}

void checkNewSemantics(const char* what, std::initializer_list<EffectInstance> effects) {
    CAPTURE(what);
    const CommandList cl = glassList(effects);
    const Drawn masked = draw(cl, true);
    const Drawn legacy = draw(cl, false);
    CHECK(masked.masked == 1u);
    CHECK(legacy.masked == 0u);
    CHECK(masked.passes == legacy.passes);  // a third input, not a pass
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH);
    CHECK(matches(masked.px, sw, kW, kH, what));
    golden::GoldenResult old = golden::compareRgba(legacy.px.data(), sw.data(), kW, kH, 16, 1.0f);
    std::printf("wgpu backdrop mask %s: legacy semantics vs sw %u px differ, max %u; %zu px changed\n", what,
                old.diffPixels, old.maxChannelDiff, differing(masked.px, legacy.px));
}

void checkUnchanged(const char* what, std::initializer_list<EffectInstance> effects) {
    CAPTURE(what);
    const CommandList cl = glassList(effects);
    const Drawn masked = draw(cl, true);
    const Drawn legacy = draw(cl, false);
    CHECK(masked.masked == 0u);
    CHECK(masked.passes == legacy.passes);
    CHECK(differing(masked.px, legacy.px) == 0u);
    CHECK(matches(masked.px, renderSw(cl, kW, kH), kW, kH, what));
}

}  // namespace

TEST_CASE("wgpu backdrop mask: shadows under a glass panel follow the new semantics and match SW (T-729)") {
    checkNewSemantics("outer shadow", {blurFx(), outerFx()});
    checkNewSemantics("inset shadow", {blurFx(), insetFx()});
    checkNewSemantics("glow", {blurFx(), glowFx(false)});
    checkNewSemantics("inner glow", {blurFx(), glowFx(true)});
    checkNewSemantics("outer and inset", {blurFx(), outerFx(), insetFx()});
}

TEST_CASE("wgpu backdrop mask: other stacks are byte-identical with and without the capability (T-729)") {
    checkUnchanged("plain backdrop blur", {blurFx()});
    checkUnchanged("backdrop blur then color adjust", {blurFx(), adjustFx()});
    checkUnchanged("backdrop blur, shadow, color adjust", {blurFx(), outerFx(), adjustFx()});
    checkUnchanged("backdrop blur not first", {outerFx(), blurFx()});
}
