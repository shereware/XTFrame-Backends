// T-730 (Docs/design/10-backends.md §10.18 "B 定案" second step, Docs/tasks/T-730.md "測試"): through
// xt_wgpu, a glass panel served from its cached pair (the source layer S and the shadow result R') next
// to the same frame drawn by a renderer that sees the list for the first time: byte for byte at rest,
// under a backdrop that moves, and a whole number of pixels away; a fraction of a pixel away within the
// T-721 tolerance, then byte for byte once it settles.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 256, kH = 192;

EffectInstance on(EffectInstance e) {
    e.enabled = 1;
    return e;
}

struct Panel {
    Vec2 at{30, 20};    ///< where its effect bounds start
    float stripes = 0;  ///< the backdrop's offset
    Color content{1, 1, 1, 0.5f};
    float shadowBlur = 12;
};

/// Hard-edged stripes, then a translucent glass panel (BackdropBlur, outer and inset shadows) laid out
/// as the Painter does: its world transform on the fills and on the effect, its bounds on the screen.
CommandList glassList(const Panel& p) {
    CommandList cl = makeList(kW, kH);
    for (int i = 0; i < 10; ++i) {
        addFillRect(cl, p.stripes + static_cast<float>(i) * 32.0f - 32.0f, 0, 32, kH,
                    Color{0.1f * static_cast<float>(i), 0.35f, 0.9f - 0.08f * static_cast<float>(i), 1});
    }
    const float margin = 30, w = 128, h = 80;
    const Mat2x3 world = translation(Vec2{p.at.x + margin, p.at.y + margin});
    addPushEffect(
        cl,
        {on(effect(EffectKind::BackdropBlur, {10, 0.4f, 0.1f})),
         on(effect(EffectKind::DropShadow, {3, 7, p.shadowBlur, 0, 1, 0}, Color{0.05f, 0, 0.1f, 0.75f})),
         on(effect(EffectKind::DropShadow, {0, 3, 8, 0, 1, 1}, Color{0, 0, 0.2f, 0.6f}))},
        Rect{p.at.x, p.at.y, w + 2 * margin, h + 2 * margin}, Rect{p.at.x + margin, p.at.y + margin, w, h});
    cl.effectCmds.back().xf = world;
    addFill(cl, addPolygon(cl, {Vec2{0, 0}, Vec2{w, 0}, Vec2{w, h}, Vec2{0, h}}), p.content,
            FillRule::NonZero, world);
    addFill(cl, addPolygon(cl, {Vec2{26.3f, 12.6f}, Vec2{106.2f, 22.1f}, Vec2{56.7f, 70.4f}}),
            Color{0.2f, 0.9f, 0.6f, 0.65f}, FillRule::NonZero, world);
    addPopEffect(cl);
    return cl;
}

std::vector<uint8_t> uncached(const CommandList& cl) {
    Session fresh(kW, kH);
    const std::vector<uint8_t> px = fresh.render(cl);
    CHECK(fresh.renderer.stats().backdropPairHits == 0u);
    CHECK_MESSAGE(fresh.error.empty(), fresh.error);
    return px;
}

struct Diff {
    int maxChannel = 0;
    double mean = 0;
};

/// T-733: a CPU adapter (lavapipe, WARP: WGPUAdapterType_CPU in the shared adapter's info) may
/// differ by 1 in a channel where a hardware adapter is byte for byte. Printed once.
bool cpuAdapter() {
    static const bool cpu = [] {
        WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
        const WGPUAdapter adapter = xt::wgpu::test::sharedInit().adapter;
        bool isCpu = false;
        if (adapter != nullptr && wgpuAdapterGetInfo(adapter, &info) == WGPUStatus_Success) {
            isCpu = info.adapterType == WGPUAdapterType_CPU;
            wgpuAdapterInfoFreeMembers(info);
        }
        std::printf("wgpu backdrop cache: %s adapter\n", isCpu ? "CPU" : "hardware");
        return isCpu;
    }();
    return cpu;
}

Diff diffOf(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    Diff d;
    double sum = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const int v = std::abs(int(a[i]) - int(b[i]));
        d.maxChannel = v > d.maxChannel ? v : d.maxChannel;
        sum += v;
    }
    d.mean = a.empty() ? 0.0 : sum / double(a.size());
    return d;
}

}  // namespace

TEST_CASE("wgpu backdrop cache: T-730 a glass panel at rest is served from its pair, byte for byte") {
    Session s(kW, kH);
    uint32_t inFull = 0;
    for (int i = 0; i < 6; ++i) {
        CAPTURE(i);
        Panel p;
        p.stripes = i >= 3 ? static_cast<float>(i) * 5.0f : 0.0f;  // later, what lies below moves
        const CommandList cl = glassList(p);
        const std::vector<uint8_t> px = s.render(cl);
        CHECK_MESSAGE(s.error.empty(), s.error);
        const wgpu::WgpuStats st = s.renderer.stats();
        if (i == 0) {
            CHECK(st.backdropPairMisses == 1u);
            inFull = st.passes;
        } else if (i == 1) {
            CHECK(st.backdropPairStores == 1u);
            CHECK(st.cacheStores == 2u);
        } else {
            CHECK(st.backdropPairHits == 1u);
            CHECK(st.backdropPairMisses == 0u);
            CHECK(st.passes < inFull);
        }
        CHECK(diffOf(px, uncached(cl)).maxChannel == 0);
    }
}

TEST_CASE("wgpu backdrop cache: T-730 another content or shadow is drawn again") {
    Session s(kW, kH);
    Panel p;
    for (int i = 0; i < 3; ++i) {
        s.render(glassList(p));
    }
    CHECK(s.renderer.stats().backdropPairHits == 1u);
    p.content = Color{1, 0.8f, 0.8f, 0.55f};
    std::vector<uint8_t> px = s.render(glassList(p));
    CHECK(s.renderer.stats().backdropPairMisses == 1u);
    CHECK(diffOf(px, uncached(glassList(p))).maxChannel == 0);
    p.shadowBlur = 18;
    px = s.render(glassList(p));
    CHECK(s.renderer.stats().backdropPairMisses == 1u);
    CHECK(diffOf(px, uncached(glassList(p))).maxChannel == 0);
}

TEST_CASE(
    "wgpu backdrop cache: T-730 a panel moving by whole pixels is served from its pair, byte for byte") {
    Session s(kW, kH);
    for (int i = 0; i < 6; ++i) {
        CAPTURE(i);
        Panel p;
        p.at = Vec2{30.0f + float(i), 20.0f + float(i / 2)};
        const CommandList cl = glassList(p);
        const std::vector<uint8_t> px = s.render(cl);
        if (i >= 2) {
            CHECK(s.renderer.stats().backdropPairHits == 1u);
        }
        // T-733 / T-735: the cached result was rasterised at the old position and pasted exactly (nearest
        // texel); the reference is rasterised afresh at the new one. Antialiased edges are translation
        // invariant bit for bit on hardware, not on a CPU adapter: there 1 in a channel is accepted.
        const std::vector<uint8_t> ref = uncached(cl);
        const Diff d = diffOf(px, ref);
        size_t differ = 0;
        for (size_t k = 0; k + 3 < px.size() && k + 3 < ref.size(); k += 4) {
            differ += (px[k] != ref[k] || px[k + 1] != ref[k + 1] || px[k + 2] != ref[k + 2] ||
                       px[k + 3] != ref[k + 3])
                          ? 1u
                          : 0u;
        }
        std::printf("wgpu backdrop cache whole-pixel move %d: %zu px differ, max channel %d\n", i, differ,
                    d.maxChannel);
        CHECK(d.maxChannel <= (cpuAdapter() ? 1 : 0));
    }
}

TEST_CASE("wgpu backdrop cache: T-730 a panel moving by fractions is pasted bilinearly and settles at rest") {
    Session s(kW, kH);
    const float xs[] = {30.0f, 30.25f, 30.5f, 30.75f, 31.125f};
    Panel p;
    int frame = 0;
    for (float x : xs) {
        CAPTURE(frame);
        p.at = Vec2{x, 20.0f + 0.375f * float(frame)};
        const CommandList cl = glassList(p);
        const std::vector<uint8_t> px = s.render(cl);
        if (frame >= 2) {
            CHECK(s.renderer.stats().backdropPairHits == 1u);
            CHECK(s.renderer.stats().effectCachePhaseHits == 1u);
            const Diff d = diffOf(px, uncached(cl));
            std::printf("T-730 phase hit, frame %d: max channel diff %d, mean %.4f\n", frame, d.maxChannel,
                        d.mean);
            CHECK(d.maxChannel <= 64);
            CHECK(d.mean <= 0.25);
        }
        ++frame;
    }
    const CommandList rest = glassList(p);
    const std::vector<uint8_t> reference = uncached(rest);
    const std::vector<uint8_t> settled = s.render(rest);
    CHECK(s.renderer.stats().backdropPairMisses == 1u);
    CHECK(s.renderer.stats().effectCacheSettles == 1u);
    CHECK(diffOf(settled, reference).maxChannel == 0);
    for (int i = 0; i < 2; ++i) {
        const std::vector<uint8_t> px = s.render(rest);
        CHECK(s.renderer.stats().backdropPairHits == 1u);
        CHECK(diffOf(px, reference).maxChannel == 0);
    }
}

TEST_CASE("wgpu backdrop cache: T-730 a panel cut by the surface edge hits with its absolute key") {
    Session s(kW, kH);
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i);
        Panel p;
        p.at = Vec2{-40, 20};
        p.stripes = float(i) * 3.0f;
        const CommandList cl = glassList(p);
        const std::vector<uint8_t> px = s.render(cl);
        if (i >= 2) {
            CHECK(s.renderer.stats().backdropPairHits == 1u);
        }
        CHECK(diffOf(px, uncached(cl)).maxChannel == 0);
    }
}
