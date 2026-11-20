// T-754 (Docs/design/10-backends.md §10.20, Docs/tasks/T-754.md "測試"): a layer or effect whose
// bounds are known to lie wholly outside the viewport (boundsKnown, an empty clamped hint) is not
// drawn -- the same pixels as the whole-surface fallback it used to get (an unknown, empty hint),
// in far fewer passes, and the same as xt::sw.

#include <cstdint>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

/// A visible card with a drop shadow, then three sections wholly outside the 96 x 80 viewport --
/// a shadowed card left of it, an opacity layer above it and a glass panel right of it -- whose
/// clamped hints are empty; `known` says whether those hints are known (T-754) or the old
/// "unknown, empty" fallback.
CommandList offscreenScene(int w, int h, bool known) {
    CommandList cl = makeList(w, h);
    addFillRect(cl, 0, 0, static_cast<float>(w), static_cast<float>(h), Color{0.9f, 0.85f, 0.7f, 1});
    EffectInstance shadow = effect(EffectKind::DropShadow, {3, 3, 6, 0, 1, 0}, Color{0, 0, 0, 0.7f});
    shadow.enabled = 1;
    addPushEffect(cl, {shadow}, Rect{10, 10, 52, 46}, Rect{20, 20, 32, 26});
    cl.effectCmds.back().boundsKnown = 1;
    addFillRect(cl, 20, 20, 32, 26, Color{0.2f, 0.4f, 0.9f, 1});
    addPopEffect(cl);

    addPushEffect(cl, {shadow}, Rect{0, 30, 0, 46}, Rect{-80, 40, 40, 26});
    cl.effectCmds.back().boundsKnown = known ? 1 : 0;
    addFillRect(cl, -80, 40, 40, 26, Color{1, 0, 0, 1});
    addPopEffect(cl);

    addPushLayer(cl, 0.5f, BlendMode::Normal, Rect{10, 0, 50, 0});
    cl.layers.back().boundsKnown = known ? 1 : 0;
    addFillRect(cl, 10, -60, 50, 30, Color{0, 1, 0, 1});
    addPopLayer(cl);

    EffectInstance glass = effect(EffectKind::BackdropBlur, {8, 1, 1});
    glass.enabled = 1;
    addPushEffect(cl, {glass}, Rect{static_cast<float>(w), 10, 0, 40},
                  Rect{static_cast<float>(w) + 10, 10, 40, 40},
                  /*sceneRead=*/true);
    cl.effectCmds.back().boundsKnown = known ? 1 : 0;
    addFillRect(cl, static_cast<float>(w) + 10, 10, 40, 40, Color{1, 1, 1, 0.4f});
    addPopEffect(cl);
    return cl;
}

}  // namespace

TEST_CASE("wgpu offscreen cull: sections with known, empty bounds are not drawn (T-754)") {
    constexpr int kW = 96, kH = 80;
    const CommandList before = offscreenScene(kW, kH, false);
    const CommandList after = offscreenScene(kW, kH, true);

    Session old(kW, kH);
    const std::vector<uint8_t> oldPx = old.render(before);
    CHECK_MESSAGE(old.error.empty(), old.error);
    const uint32_t oldPasses = old.renderer.stats().passes;

    Session now(kW, kH);
    const std::vector<uint8_t> newPx = now.render(after);
    CHECK_MESSAGE(now.error.empty(), now.error);
    const uint32_t newPasses = now.renderer.stats().passes;
    MESSAGE("passes: whole-surface fallback " << oldPasses << ", culled " << newPasses);
    CHECK(newPasses < oldPasses);
    // The culled sections drew nothing inside the viewport before either: the same bytes.
    CHECK(newPx == oldPx);
    CHECK(matches(newPx, renderSw(after, kW, kH), kW, kH, "offscreen cull"));
    CHECK(renderSw(after, kW, kH) == renderSw(before, kW, kH));
}
