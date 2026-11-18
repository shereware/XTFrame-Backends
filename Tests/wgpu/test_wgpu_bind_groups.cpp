// T-726 (Docs/tasks/T-726.md item B): views destroyed by a WgpuSharedCache reach every device through its
// view log; each device removes exactly the bind groups naming them (one device's own destroyed views:
// test_wgpu_device.cpp).

#include <cstdint>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/wgpu/wgpu_device.h"
#include "xt/wgpu/wgpu_shared.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

/// A layer and a blurred box: frames with transients (Composite / Effect groups name their views).
CommandList layeredList(int w, int h) {
    CommandList cl = makeList(w, h);
    addFillRect(cl, 0, 0, static_cast<float>(w), static_cast<float>(h), Color{0.1f, 0.2f, 0.3f, 1});
    addPushLayer(cl, 0.6f, BlendMode::Normal, Rect{8, 8, 40, 40});
    addFillRect(cl, 8, 8, 30, 30, Color{1, 0, 0, 1});
    addFillRect(cl, 20, 20, 28, 28, Color{0, 0, 1, 1});
    addPopLayer(cl);
    addPushEffect(cl, {effect(EffectKind::Blur, {2, 2})}, Rect{36, 36, 26, 26}, Rect{40, 40, 18, 18});
    addFillRect(cl, 40, 40, 18, 18, Color{0, 1, 0, 1});
    addPopEffect(cl);
    return cl;
}

}  // namespace

TEST_CASE("wgpu: views a shared cache destroys leave no bind group on any device (T-726)") {
    constexpr int kW = 64, kH = 64;
    wgpu::WgpuSharedCache cache;
    wgpu::WgpuInit in = sharedInit();
    in.error.clear();
    in.shared = &cache;
    const CommandList cl = layeredList(kW, kH);
    struct Dev {
        wgpu::WgpuRenderer r;
        std::vector<uint8_t> first;
        uint32_t live = 0;
    } devs[2];
    DrawTarget target(kW, kH);
    for (Dev& d : devs) {
        REQUIRE(d.r.init(in));
        d.r.resize(kW, kH);
        d.r.setClearColor(0, 0, 0, 1);
        d.r.render(cl, nullptr, nullptr, target.view);
        REQUIRE_MESSAGE(std::string(d.r.lastError()).empty(), d.r.lastError());
        REQUIRE(d.r.device()->sharing());
        d.first.resize(static_cast<size_t>(kW) * kH * 4);
        REQUIRE(d.r.readback(target.texture, d.first));
        d.live = d.r.device()->lastStats().liveBindings;
    }
    CHECK(devs[0].first == devs[1].first);
    for (int round = 0; round < 3; ++round) {
        // Every pooled transient goes (their views are logged); each device drops the groups naming
        // them before its next submit, draws the same pixels with new transients, and holds no more
        // bindings than before.
        CHECK(cache.trimPools() > 0u);
        for (Dev& d : devs) {
            d.r.render(cl, nullptr, nullptr, target.view);
            REQUIRE_MESSAGE(std::string(d.r.lastError()).empty(), d.r.lastError());
            const wgpu::WgpuDeviceStats& s = d.r.device()->lastStats();
            CHECK(s.bindGroupEvictions > 0u);
            CHECK(s.bindGroups > 0u);
            CHECK(s.liveBindings == d.live);
            std::vector<uint8_t> px(static_cast<size_t>(kW) * kH * 4);
            REQUIRE(d.r.readback(target.texture, px));
            CHECK(px == d.first);
        }
        // Nothing destroyed: everything is cached, nothing goes.
        for (Dev& d : devs) {
            d.r.render(cl, nullptr, nullptr, target.view);
            const wgpu::WgpuDeviceStats& s = d.r.device()->lastStats();
            CHECK(s.bindGroupEvictions == 0u);
            CHECK(s.liveBindings == d.live);
        }
    }
}
