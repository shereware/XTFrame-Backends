// T-734 (Docs/design/10-backends.md §10.19): GPU timestamps are a diagnostic that is off by default --
// the same pixels with and without them, no query set while off, and per-frame results that match the
// passes the frame encoded. SKIPs without an adapter that has WGPUFeatureName_TimestampQuery.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/wgpu/wgpu_renderer.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

/// A layer and a blurred box: several passes per frame.
CommandList timedList(int w, int h) {
    CommandList cl = makeList(w, h);
    addFillRect(cl, 0, 0, static_cast<float>(w), static_cast<float>(h), Color{0.1f, 0.2f, 0.3f, 1});
    addPushLayer(cl, 0.6f, BlendMode::Normal, Rect{8, 8, 40, 40});
    addFillRect(cl, 8, 8, 30, 30, Color{1, 0, 0, 1});
    addPopLayer(cl);
    addPushEffect(cl, {effect(EffectKind::Blur, {2, 2})}, Rect{36, 36, 26, 26}, Rect{40, 40, 18, 18});
    addFillRect(cl, 40, 40, 18, 18, Color{0, 1, 0, 1});
    addPopEffect(cl);
    return cl;
}

struct OwnTarget {
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    OwnTarget(WGPUDevice device, int w, int h) {
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.usage =
            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
        d.size = WGPUExtent3D{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        texture = wgpuDeviceCreateTexture(device, &d);
        view = wgpuTextureCreateView(texture, nullptr);
    }
    ~OwnTarget() {
        wgpuTextureViewRelease(view);
        wgpuTextureDestroy(texture);
        wgpuTextureRelease(texture);
    }
    OwnTarget(const OwnTarget&) = delete;
    OwnTarget& operator=(const OwnTarget&) = delete;
};

}  // namespace

TEST_CASE("wgpu: GPU timestamps leave the frame unchanged and report the frame's passes (T-734)") {
    constexpr int kW = 64, kH = 64;
    wgpu::WgpuHeadlessOptions options;
    options.gpuTiming = true;
    wgpu::WgpuInit in = wgpu::WgpuInit::headless(options);
    if (in.device == nullptr) {
        MESSAGE("SKIP: no WebGPU device with gpuTiming (" << in.error << ")");
        return;
    }
    const CommandList cl = timedList(kW, kH);
    OwnTarget target(in.device, kW, kH);
    std::vector<uint8_t> off(static_cast<size_t>(kW) * kH * 4), on(off.size());
    {
        wgpu::WgpuRenderer plain;
        REQUIRE(plain.init(in));
        plain.resize(kW, kH);
        plain.setClearColor(0, 0, 0, 1);
        CHECK_FALSE(plain.gpuTimingEnabled());
        plain.render(cl, nullptr, nullptr, target.view);
        REQUIRE_MESSAGE(std::string(plain.lastError()).empty(), plain.lastError());
        CHECK_FALSE(plain.gpuTimingHasQuerySet());  // off: no query set, nothing extra encoded
        REQUIRE(plain.readback(target.texture, off));
        plain.shutdown();
    }
    wgpu::WgpuRenderer r;
    REQUIRE(r.init(in));
    r.resize(kW, kH);
    r.setClearColor(0, 0, 0, 1);
    if (!r.setGpuTiming(true)) {
        CHECK_FALSE(r.gpuTimingEnabled());
        CHECK_FALSE(r.gpuTimingHasQuerySet());
        MESSAGE("SKIP: the adapter has no TimestampQuery (" << in.notes << ")");
        r.shutdown();
        in.release();
        return;
    }
    CHECK(r.gpuTimingEnabled());
    std::map<uint64_t, uint32_t> passesOf;
    std::vector<wgpu::WgpuGpuTiming> results;
    for (uint64_t f = 0; f < 10; ++f) {
        r.render(cl, nullptr, nullptr, target.view);
        REQUIRE_MESSAGE(std::string(r.lastError()).empty(), r.lastError());
        passesOf[f] = r.stats().passes;
        if (f == 0) {
            REQUIRE(r.readback(target.texture, on));
            CHECK(on == off);  // the same pixels with timestamps
        }
        wgpuDevicePoll(in.device, 1, nullptr);
        r.takeGpuTimings(results);
    }
    CHECK(r.gpuTimingHasQuerySet());
    REQUIRE_FALSE(results.empty());
    uint64_t last = 0;
    for (size_t i = 0; i < results.size(); ++i) {
        const wgpu::WgpuGpuTiming& t = results[i];
        CAPTURE(t.frame);
        if (i > 0) CHECK(t.frame > last);  // oldest first
        last = t.frame;
        REQUIRE(passesOf.count(t.frame) == 1u);
        CHECK(t.passes == passesOf[t.frame]);
        CHECK(t.spanMs > 0.0);
        if (t.encoderLevel) CHECK(t.passSumMs <= t.spanMs * 1.05);
    }
    std::printf(
        "T-734: %zu timed frames, encoder level %d, last span %.4f ms, pass sum %.4f ms, dropped %llu\n",
        results.size(), results.back().encoderLevel ? 1 : 0, results.back().spanMs, results.back().passSumMs,
        static_cast<unsigned long long>(r.gpuTimingDropped()));
    CHECK(r.setGpuTiming(false));
    CHECK_FALSE(r.gpuTimingEnabled());
    CHECK_FALSE(r.gpuTimingHasQuerySet());
    r.render(cl, nullptr, nullptr, target.view);
    wgpuDevicePoll(in.device, 1, nullptr);
    results.clear();
    CHECK(r.takeGpuTimings(results) == 0u);
    r.shutdown();
    in.release();
}
