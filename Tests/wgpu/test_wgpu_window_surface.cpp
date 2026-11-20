// T-758 (Docs/tasks/T-758.md, Docs/design/10-backends.md §10.20 2026-11-18 note (1)): a window surface
// is sampled by a frame that reads target 0 (a mix-blend-mode backdrop copy, a backdrop blur pyramid, a
// 4x surface loaded without a clear). chooseSurfaceUsage asks for TEXTURE_BINDING when the surface offers
// it; when it does not, WgpuWindowSurface draws into an intermediate texture and moves it to the surface
// at present. The surface is stood in for by an offscreen texture of the usage a surface may offer
// (WgpuWindowSurface::createForTesting); the frames match the headless target byte for byte.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/wgpu/wgpu_window.h"

using namespace xt;
using namespace xt::wgpu::test;
using xt::wgpu::chooseSurfaceUsage;
using xt::wgpu::SurfaceUsageChoice;
using xt::wgpu::WgpuWindowSurface;

namespace {

constexpr int kW = 160, kH = 120;

constexpr WGPUTextureUsage kRA = WGPUTextureUsage_RenderAttachment;
constexpr WGPUTextureUsage kSrc = WGPUTextureUsage_CopySrc;
constexpr WGPUTextureUsage kDst = WGPUTextureUsage_CopyDst;
constexpr WGPUTextureUsage kBind = WGPUTextureUsage_TextureBinding;

EffectInstance on(EffectInstance e) {
    e.enabled = 1;
    return e;
}

/// Hard-edged stripes the frames below read back.
void addStripes(CommandList& cl) {
    for (int i = 0; i < 6; ++i) {
        addFillRect(cl, static_cast<float>(i) * 28.0f, 0, 28, kH,
                    Color{0.15f * static_cast<float>(i), 0.4f, 0.9f - 0.12f * static_cast<float>(i), 1});
    }
}

/// A top-level panel with a backdrop blur (its pyramid is built from target 0).
CommandList backdropBlurList() {
    CommandList cl = makeList(kW, kH);
    addStripes(cl);
    addPushEffect(cl, {on(effect(EffectKind::BackdropBlur, {6, 0.3f, 0.1f}))}, Rect{30, 20, 100, 70},
                  Rect{30, 20, 100, 70});
    addFillRect(cl, 30, 20, 100, 70, Color{1, 1, 1, 0.35f});
    addPopEffect(cl);
    return cl;
}

/// A top-level layer with mix-blend-mode: multiply (its backdrop is copied from target 0).
CommandList blendList() {
    CommandList cl = makeList(kW, kH);
    addStripes(cl);
    addPushLayer(cl, 0.9f, BlendMode::Multiply, Rect{20, 30, 110, 60});
    addFillRect(cl, 20, 30, 110, 60, Color{0.9f, 0.6f, 0.2f, 1});
    addPopLayer(cl);
    return cl;
}

/// A plain frame (the 4x no-clear case reads target 0 before any command).
CommandList plainList() {
    CommandList cl = makeList(kW, kH);
    addFillRect(cl, 16, 16, 64, 48, Color{1, 0, 0, 1});
    addFill(cl, addPolygon(cl, {Vec2{90, 10}, Vec2{150, 100}, Vec2{70, 110}}), Color{0.2f, 0.9f, 0.4f, 0.7f});
    return cl;
}

/// A texture standing in for a window's swapchain texture: `usage`, RGBA8Unorm, kW x kH.
struct StandIn {
    WGPUTexture texture = nullptr;
    explicit StandIn(WGPUTextureUsage usage) {
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.usage = usage;
        d.size = WGPUExtent3D{static_cast<uint32_t>(kW), static_cast<uint32_t>(kH), 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        texture = wgpuDeviceCreateTexture(sharedInit().device, &d);
    }
    ~StandIn() {
        wgpuTextureDestroy(texture);
        wgpuTextureRelease(texture);
    }
    StandIn(const StandIn&) = delete;
    StandIn& operator=(const StandIn&) = delete;
};

enum class Clear { Yes, No };

/// The frame drawn by a renderer into the headless target (TEXTURE_BINDING | COPY_SRC | COPY_DST).
std::vector<uint8_t> headless(const CommandList& cl, uint32_t samples, Clear clear) {
    Session s(kW, kH, samples);
    if (clear == Clear::No) {
        s.renderer.disableClear();
        s.renderer.render(cl, nullptr, nullptr, s.target.view);
        s.error = s.renderer.lastError();
    } else {
        s.render(cl);
    }
    CHECK_MESSAGE(s.error.empty(), s.error);
    // The frame does read target 0: a copy pass from it, or (4x without a clear) the load blit.
    const xt::wgpu::WgpuDeviceStats& st = s.renderer.device()->lastStats();
    if (clear == Clear::No) {
        CHECK(st.blits >= 1u);
    } else {
        CHECK(st.surfaceReads >= 1u);
    }
    return s.read();
}

/// The frame drawn through a WgpuWindowSurface standing for a texture of `usage`, read back from that
/// texture after present (so the transfer from an intermediate texture is in it). `error` gets the
/// renderer's error (device errors included).
std::vector<uint8_t> throughSurface(const CommandList& cl, uint32_t samples, Clear clear,
                                    WGPUTextureUsage usage, bool& readable, std::string& error) {
    StandIn standIn(usage);
    WgpuWindowSurface surface;
    std::string createError;
    REQUIRE_MESSAGE(surface.createForTesting(sharedInit().device, standIn.texture, createError), createError);
    readable = surface.surfaceReadable();
    xt::wgpu::WgpuInit in = sharedInit();
    in.sampleCount = samples;
    in.error.clear();
    xt::wgpu::WgpuRenderer renderer;
    REQUIRE(renderer.init(in));
    renderer.resize(kW, kH);
    if (clear == Clear::No) {
        renderer.disableClear();
    } else {
        renderer.setClearColor(0, 0, 0, 1);
    }
    REQUIRE(surface.acquire());
    REQUIRE(surface.frameCopyable());
    renderer.render(cl, nullptr, nullptr, surface.frameView());
    error = renderer.lastError();
    surface.present();
    std::vector<uint8_t> px(static_cast<size_t>(kW) * static_cast<size_t>(kH) * 4);
    REQUIRE_MESSAGE(renderer.readback(standIn.texture, px), renderer.lastError());
    const std::string after = renderer.lastError();
    if (error.empty()) error = after;
    renderer.shutdown();
    surface.release();
    return px;
}

}  // namespace

TEST_CASE("wgpu window surface: T-758 chooseSurfaceUsage asks for TEXTURE_BINDING when offered") {
    SurfaceUsageChoice c = chooseSurfaceUsage(kRA | kSrc | kDst | kBind);
    CHECK(c.readable);
    CHECK(c.usage == (kRA | kSrc | kBind));  // COPY_DST only for the fallback

    c = chooseSurfaceUsage(kRA | kBind);
    CHECK(c.readable);
    CHECK(c.usage == (kRA | kBind));

    c = chooseSurfaceUsage(kRA | kSrc);
    CHECK_FALSE(c.readable);
    CHECK(c.usage == (kRA | kSrc));

    c = chooseSurfaceUsage(kRA | kSrc | kDst);  // the intermediate is copied in
    CHECK_FALSE(c.readable);
    CHECK(c.usage == (kRA | kSrc | kDst));

    c = chooseSurfaceUsage(kRA);
    CHECK_FALSE(c.readable);
    CHECK(c.usage == kRA);

    c = chooseSurfaceUsage(WGPUTextureUsage_None);  // RENDER_ATTACHMENT is always asked for
    CHECK_FALSE(c.readable);
    CHECK(c.usage == kRA);
}

TEST_CASE("wgpu window surface: T-758 frames that read target 0 match the headless target byte for byte") {
    struct Case {
        const char* name;
        CommandList list;
        uint32_t samples;
        Clear clear;
    };
    const Case cases[] = {
        {"top-level backdrop blur", backdropBlurList(), 1, Clear::Yes},
        {"top-level mix-blend-mode", blendList(), 1, Clear::Yes},
        {"4x without a clear colour", plainList(), 4, Clear::No},
        {"4x backdrop blur", backdropBlurList(), 4, Clear::Yes},
    };
    // RENDER_ATTACHMENT | COPY_SRC: blitted from the intermediate; + COPY_DST: copied from it; +
    // TEXTURE_BINDING: drawn into directly.
    const WGPUTextureUsage usages[] = {kRA | kSrc, kRA | kSrc | kDst, kRA | kSrc | kBind};
    for (const Case& c : cases) {
        const std::vector<uint8_t> expected = headless(c.list, c.samples, c.clear);
        for (WGPUTextureUsage usage : usages) {
            CAPTURE(c.name);
            CAPTURE(static_cast<uint64_t>(usage));
            bool readable = false;
            std::string error;
            const std::vector<uint8_t> got =
                throughSurface(c.list, c.samples, c.clear, usage, readable, error);
            CHECK(readable == ((usage & kBind) != 0));
            CHECK_MESSAGE(error.empty(), error);
            CHECK(got == expected);
        }
    }
}

TEST_CASE("wgpu window surface: T-758 the intermediate follows the surface's size") {
    // A second stand-in of another size behind the same WgpuWindowSurface object (as a resized window
    // gives textures of another size): the intermediate is made again at the new size, and the frame
    // still reaches the surface.
    WgpuWindowSurface surface;
    xt::wgpu::WgpuRenderer renderer;
    REQUIRE(renderer.init(sharedInit()));
    renderer.setClearColor(0, 0, 1, 1);
    for (int pass = 0; pass < 2; ++pass) {
        const int w = pass == 0 ? kW : 96, h = pass == 0 ? kH : 64;
        WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
        d.usage = kRA | kSrc;
        d.size = WGPUExtent3D{static_cast<uint32_t>(w), static_cast<uint32_t>(h), 1};
        d.format = WGPUTextureFormat_RGBA8Unorm;
        WGPUTexture standIn = wgpuDeviceCreateTexture(sharedInit().device, &d);
        std::string err;
        REQUIRE(surface.createForTesting(sharedInit().device, standIn, err));
        renderer.resize(w, h);
        for (int frame = 0; frame < 2; ++frame) {
            REQUIRE(surface.acquire());
            CHECK(wgpuTextureGetWidth(surface.frameTexture()) == static_cast<uint32_t>(w));
            CHECK(wgpuTextureGetHeight(surface.frameTexture()) == static_cast<uint32_t>(h));
            renderer.render(makeList(w, h), nullptr, nullptr, surface.frameView());
            surface.present();
        }
        std::vector<uint8_t> px(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
        REQUIRE(renderer.readback(standIn, px));
        CHECK(std::string(renderer.lastError()).empty());
        CHECK(pixel(px, w, w - 1, h - 1)[2] == 255);
        CHECK(pixel(px, w, 0, 0)[2] == 255);
        wgpuTextureDestroy(standIn);  // the surface holds no reference outside acquire / present
        wgpuTextureRelease(standIn);
    }
    renderer.shutdown();
    surface.release();
}

TEST_CASE("wgpu window surface: T-749 SurfaceReads::None draws into the surface whatever it offers") {
    using xt::wgpu::SurfaceReads;
    SurfaceUsageChoice c = chooseSurfaceUsage(kRA | kSrc, SurfaceReads::None);  // D3D12's surface
    CHECK_FALSE(c.readable);
    CHECK(c.direct);
    CHECK(c.usage == (kRA | kSrc));
    c = chooseSurfaceUsage(kRA | kSrc | kDst, SurfaceReads::None);  // COPY_DST is never needed then
    CHECK(c.direct);
    CHECK(c.usage == (kRA | kSrc));
    c = chooseSurfaceUsage(kRA | kSrc | kBind, SurfaceReads::None);  // as with Sampled
    CHECK(c.readable);
    CHECK(c.direct);
    CHECK(c.usage == (kRA | kSrc | kBind));
    c = chooseSurfaceUsage(kRA | kSrc);  // the default stays T-758's
    CHECK_FALSE(c.direct);
    CHECK(chooseSurfaceUsage(kRA | kSrc | kBind).direct);
}

TEST_CASE("wgpu window surface: T-749 the C ABI target's surface makes no intermediate, same bytes") {
    // The C ABI GPU target creates its window surface with SurfaceReads::None (its composer draws
    // textured quads over target 0 and never samples it). On a surface without TEXTURE_BINDING (D3D12's
    // RenderAttachment | CopySrc, with or without CopyDst) the frame then goes into the surface texture
    // itself -- no intermediate texture, no transfer at present -- and the bytes equal those of the
    // T-758 path (the intermediate) and of the headless target.
    using xt::wgpu::SurfaceReads;
    const CommandList list = plainList();
    Session ref(kW, kH, 1);
    const std::vector<uint8_t> reference = ref.render(list);
    REQUIRE_MESSAGE(ref.error.empty(), ref.error);
    const WGPUTextureUsage usages[] = {kRA | kSrc, kRA | kSrc | kDst, kRA | kSrc | kBind};
    for (WGPUTextureUsage usage : usages) {
        CAPTURE(static_cast<uint64_t>(usage));
        std::vector<uint8_t> got[2];
        for (int pass = 0; pass < 2; ++pass) {
            const SurfaceReads reads = pass == 0 ? SurfaceReads::Sampled : SurfaceReads::None;
            StandIn standIn(usage);
            WgpuWindowSurface surface;
            std::string err;
            REQUIRE(surface.createForTesting(sharedInit().device, standIn.texture, err, reads));
            xt::wgpu::WgpuRenderer renderer;
            REQUIRE(renderer.init(sharedInit()));
            renderer.resize(kW, kH);
            renderer.setClearColor(0, 0, 0, 1);
            REQUIRE(surface.acquire());
            const bool direct = surface.frameTexture() == standIn.texture;
            CHECK(direct == surface.drawsDirectly());
            if (reads == SurfaceReads::None || (usage & kBind) != 0) {
                CHECK(direct);  // no intermediate texture
            } else {
                CHECK_FALSE(direct);  // T-758's intermediate
            }
            REQUIRE(surface.frameCopyable());
            renderer.render(list, nullptr, nullptr, surface.frameView());
            CHECK(std::string(renderer.lastError()).empty());
            surface.present();
            got[pass].resize(static_cast<size_t>(kW) * static_cast<size_t>(kH) * 4);
            REQUIRE_MESSAGE(renderer.readback(standIn.texture, got[pass]), renderer.lastError());
            CHECK(surface.usageNote().find(direct ? "frames drawn into the surface" : "intermediate") !=
                  std::string::npos);
            renderer.shutdown();
            surface.release();
        }
        CHECK(got[1] == got[0]);
        CHECK(got[1] == reference);
    }
}
