// T-721 (Docs/design/10-backends.md §10.16 W2, Docs/tasks/T-721.md "測試"): the effect result cache
// keyed independently of the segment's position, through xt_wgpu -- what a moving segment served
// from its cached result looks like next to the same frame drawn without a cache (a renderer that
// sees the list for the first time) and next to xt::sw.

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

/// A card with a drop shadow, its effect bounds starting at `at`, laid out as the Painter does (the
/// node's world transform on the fill and the effect, the bounds on the screen). A second, smaller
/// fill gives the content an inner edge.
CommandList cardList(Vec2 at) {
    CommandList cl = makeList(kW, kH);
    const float margin = 24, side = 48;
    const Mat2x3 world = translation(Vec2{at.x + margin, at.y + margin});
    addFillRect(cl, 0, 0, kW, kH, Color{0.15f, 0.2f, 0.3f, 1});  // something under the shadow
    addPushEffect(cl, {effect(EffectKind::DropShadow, {5, 5, 6, 0, 0.9f})},
                  Rect{at.x, at.y, side + 2 * margin, side + 2 * margin},
                  Rect{at.x + margin, at.y + margin, side, side});
    cl.effectCmds.back().xf = world;
    addFill(cl, addPolygon(cl, {Vec2{0, 0}, Vec2{side, 0}, Vec2{side, side}, Vec2{0, side}}),
            Color{1, 0.3f, 0.1f, 1}, FillRule::NonZero, world);
    addFill(cl, addPolygon(cl, {Vec2{10, 10}, Vec2{30, 10}, Vec2{30, 30}, Vec2{10, 30}}),
            Color{0.1f, 0.4f, 1, 0.7f}, FillRule::NonZero, world);
    addPopEffect(cl);
    return cl;
}

/// `cl` drawn by a renderer with an empty cache: the segment in full.
std::vector<uint8_t> uncached(const CommandList& cl) {
    Session fresh(kW, kH);
    const std::vector<uint8_t> px = fresh.render(cl);
    CHECK(fresh.renderer.stats().fxCacheHits == 0u);
    CHECK_MESSAGE(fresh.error.empty(), fresh.error);
    return px;
}

struct Diff {
    int maxChannel = 0;
    double mean = 0;  ///< over all channels of all pixels
    int differing = 0;
};

Diff diffOf(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    Diff d;
    double sum = 0;
    for (size_t i = 0; i + 3 < a.size(); i += 4) {
        bool differs = false;
        for (size_t c = 0; c < 4; ++c) {
            const int v = std::abs(int(a[i + c]) - int(b[i + c]));
            d.maxChannel = v > d.maxChannel ? v : d.maxChannel;
            sum += v;
            differs = differs || v != 0;
        }
        d.differing += differs ? 1 : 0;
    }
    d.mean = a.empty() ? 0.0 : sum / double(a.size());
    return d;
}

}  // namespace

TEST_CASE(
    "wgpu effect cache: T-721 a segment moving by whole pixels is served from its result, byte for byte") {
    Session s(kW, kH);
    for (int i = 0; i < 6; ++i) {
        CAPTURE(i);
        const CommandList cl = cardList(Vec2{60.0f + float(i), 40.0f + float(i / 2)});
        const std::vector<uint8_t> px = s.render(cl);
        CHECK_MESSAGE(s.error.empty(), s.error);
        if (i == 0) {
            CHECK(s.renderer.stats().fxCacheMisses == 1u);
            CHECK(s.renderer.stats().fxCacheStores == 0u);
        } else if (i == 1) {
            CHECK(s.renderer.stats().fxCacheStores == 1u);
        } else {
            CHECK(s.renderer.stats().fxCacheHits == 1u);
            CHECK(s.renderer.stats().fxCacheMisses == 0u);
        }
        const Diff d = diffOf(px, uncached(cl));
        CHECK(d.maxChannel == 0);
        CHECK(matches(px, renderSw(cl, kW, kH), kW, kH, "T-721 whole-pixel move"));
    }
}

TEST_CASE("wgpu effect cache: T-721 a segment moving by fractions is pasted bilinearly and settles at rest") {
    Session s(kW, kH);
    const float xs[] = {60.0f, 60.25f, 60.5f, 60.75f, 61.125f, 61.5f};
    int frame = 0;
    for (float x : xs) {
        CAPTURE(frame);
        const CommandList cl = cardList(Vec2{x, 40.0f + 0.375f * float(frame)});
        const std::vector<uint8_t> px = s.render(cl);
        CHECK_MESSAGE(s.error.empty(), s.error);
        if (frame >= 2) {
            // Served from the result drawn at frame 1's place: a bilinear copy a fraction of a pixel
            // on. Next to the frame drawn in full, only the edges of the card and of its shadow
            // differ, by less than a quarter of the range.
            CHECK(s.renderer.stats().fxCacheHits == 1u);
            CHECK(s.renderer.stats().fxCacheMisses == 0u);
            const Diff d = diffOf(px, uncached(cl));
            std::printf("T-721 phase hit, frame %d: max channel diff %d, mean %.4f, %d px differ\n", frame,
                        d.maxChannel, d.mean, d.differing);
            CHECK(d.maxChannel <= 64);
            CHECK(d.mean <= 0.25);
            CHECK(matches(px, renderSw(cl, kW, kH), kW, kH, "T-721 fractional move"));
        }
        ++frame;
    }
    // It stops: the next build draws it again in place (a settle: one miss, one store) ...
    const CommandList rest = cardList(Vec2{61.5f, 40.0f + 0.375f * 5.0f});
    const std::vector<uint8_t> reference = uncached(rest);
    const std::vector<uint8_t> settled = s.render(rest);
    CHECK(s.renderer.stats().fxCacheMisses == 1u);
    CHECK(s.renderer.stats().fxCacheStores == 1u);
    CHECK(diffOf(settled, reference).maxChannel == 0);
    // ... and from then on the cached frame is the uncached one, byte for byte.
    for (int i = 0; i < 2; ++i) {
        const std::vector<uint8_t> px = s.render(rest);
        CHECK(s.renderer.stats().fxCacheHits == 1u);
        CHECK(s.renderer.stats().fxCacheMisses == 0u);
        CHECK(diffOf(px, reference).maxChannel == 0);
    }
}
