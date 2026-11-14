// T-701 (Docs/design/33-world-layers.md §33.3-§33.4, Docs/tasks/T-701.md): world layers on a real
// device. gpu::WorldLayerBuilder draws every layer of a list painted in world-layer output into its
// own target; a layer that did not change is neither built nor drawn; the layers put back on top of
// one another (buildWorldLayerQuads with an orthographic view) are the ordinary picture.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/compile/compile.h"
#include "xt/context.h"
#include "xt/document.h"
#include "xt/element_ref.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/gpu/world_layers.h"
#include "xt/text_shaper.h"
#include "xt/wgpu/wgpu_device.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 320, kH = 200;

/// A Context over one document compiled from `css` / `body`.
struct Ui {
    NullTextShaper shaper;
    Context ctx{ContextConfig{&shaper, nullptr}};
    Document* doc = nullptr;
    std::vector<uint8_t> ir;
    float clock = 0.0f;

    Ui(const std::string& css, const std::string& body, bool layers) {
        const std::string text = "<xt><head><style>" + css + "</style></head><body>" + body + "</body></xt>";
        const compile::SourceFile file{"layers.xt", text};
        compile::CompileResult result =
            compile::compileDocumentFromStrings({&file, 1}, "layers.xt", compile::CompileOptions{});
        for (const compile::Diagnostic& d : result.diagnostics) {
            CAPTURE(d.message);
            CHECK(d.level != compile::Diagnostic::Level::Error);
        }
        REQUIRE(result.ok());
        ir = std::move(result.ir);
        Result<DocumentHandle> handle = ctx.loadDocument(IrBlob{ir.data(), ir.size()}, false);
        REQUIRE(handle.ok());
        doc = ctx.document(handle.value());
        ctx.setViewport(Rect{0, 0, float(kW), float(kH)}, 1.0f);
        ctx.setWorldLayerOutput(layers);
        ctx.update(0.0f);
    }
    void step() {
        clock += 0.016f;
        ctx.update(clock);
    }
    CommandList render() {
        CommandList list;
        ctx.render(list);
        return list;
    }
};

/// A wgpu device (1x), a layer builder on it and a target to put pictures into.
struct Rig {
    wgpu::WgpuRenderer renderer;
    gpu::WorldLayerBuilder layers;
    DrawTarget target{kW, kH};
    gpu::GpuFrame quads;

    Rig() {
        wgpu::WgpuInit in = sharedInit();
        in.sampleCount = 1;
        in.error.clear();
        REQUIRE(renderer.init(in));
        renderer.resize(kW, kH);
        layers.init(*renderer.device());
    }
    ~Rig() { layers.shutdown(); }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    void buildAndSubmit(const CommandList& list) {
        REQUIRE_MESSAGE(layers.build(list, nullptr, nullptr, nullptr), layers.lastError());
        layers.submit();
    }
    std::vector<uint8_t> readLayer(const gpu::WorldLayer& layer) {
        std::vector<uint8_t> px(size_t(layer.texWidth) * layer.texHeight * 4);
        WGPUTexture texture = renderer.device()->texture(layer.texture);
        REQUIRE(texture != nullptr);
        REQUIRE_MESSAGE(renderer.readback(texture, px), renderer.lastError());
        return px;
    }
    std::vector<uint8_t> readTarget() {
        std::vector<uint8_t> px(size_t(kW) * kH * 4);
        REQUIRE_MESSAGE(renderer.readback(target.texture, px), renderer.lastError());
        return px;
    }
    /// The ordinary picture of `list` on a transparent target.
    std::vector<uint8_t> flat(const CommandList& list) {
        renderer.setClearColor(0, 0, 0, 0);
        renderer.render(list, nullptr, nullptr, target.view);
        CHECK(std::string(renderer.lastError()).empty());
        return readTarget();
    }
    /// The layers stacked back in paint order (an orthographic view straight down -z).
    std::vector<uint8_t> restacked() {
        gpu::buildWorldLayerQuads(layers.layers(), mat4Identity(), kW, kH, /*clear=*/true, quads);
        wgpu::WgpuDevice& device = *renderer.device();
        device.resize(kW, kH);
        device.setSurfaceClear(true, WGPUColor{0, 0, 0, 0});
        device.setSurface(target.view);
        device.submit(quads);
        device.setSurface(nullptr);
        return readTarget();
    }
};

const char* kCss =
    "#root { position: absolute; left: 0px; top: 0px; width: 320px; height: 200px; "
    "background-color: #18222e; }"
    "#grid { position: absolute; left: 16px; top: 16px; width: 288px; height: 168px; "
    "border: 2px #3a5a7a; border-radius: 12px; }"
    "#panel { position: absolute; left: 64px; top: 40px; width: 160px; height: 96px; "
    "background-color: #2f6fed; border-radius: 16px; border: 3px #cfe3ff; }"
    "#chip { position: absolute; left: 20px; top: 20px; width: 60px; height: 24px; "
    "background-color: #ffd34d; border-radius: 12px; }"
    "#badge { position: absolute; left: 200px; top: 110px; width: 72px; height: 48px; "
    "background-color: rgba(255, 90, 90, 0.8); border-radius: 24px; }";
const char* kBody =
    "<div id=\"root\"><div id=\"grid\"></div><div id=\"panel\"><div id=\"chip\"></div></div>"
    "<div id=\"badge\"></div></div>";

const gpu::WorldLayer* layerOf(std::span<const gpu::WorldLayer> layers, uint32_t id) {
    for (const gpu::WorldLayer& layer : layers) {
        if (layer.id == id) return &layer;
    }
    return nullptr;
}

}  // namespace

TEST_CASE("wgpu world layers: a UI without any switch is one layer equal to the ordinary picture (T-701)") {
    Ui on(kCss, kBody, /*layers=*/true);
    Ui off(kCss, kBody, /*layers=*/false);
    Rig rig;
    rig.buildAndSubmit(on.render());
    REQUIRE(rig.layers.layers().size() == 1);
    const gpu::WorldLayer& base = rig.layers.layers()[0];
    CHECK(base.id == 0u);
    CHECK(base.texWidth == kW);
    CHECK(base.texHeight == kH);
    CHECK(base.changed == 1);
    const std::vector<uint8_t> layerPx = rig.readLayer(base);
    const std::vector<uint8_t> flatPx = rig.flat(off.render());
    const golden::GoldenResult r = golden::compareRgba(layerPx.data(), flatPx.data(), kW, kH, 0, 0.0f);
    MESSAGE("one layer vs ordinary: " << r.diffPixels << " px differ, max channel diff " << r.maxChannelDiff);
    CHECK(r.diffPixels == 0u);
}

TEST_CASE("wgpu world layers: coplanar layers stacked back are the ordinary picture (T-701)") {
    // #panel (with its child) and #badge are layers on the root's own plane; they are the last
    // things painted, so paint order is unchanged by taking them out.
    const std::string css = std::string(kCss) + "#panel { world-layer: on; } #badge { world-layer: on; }";
    Ui on(css, kBody, /*layers=*/true);
    Ui off(css, kBody, /*layers=*/false);
    Rig rig;
    rig.buildAndSubmit(on.render());
    REQUIRE(rig.layers.layers().size() == 3);
    CHECK(rig.layers.stats().built == 3u);
    CHECK(rig.layers.stats().submitted == 3u);
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), on.doc->byId("panel").node);
    REQUIRE(panel != nullptr);
    // 160 x 96 with a pixel of slack, out to the 32 px grid, at density 1.
    CHECK(panel->rect.x == -32.0f);
    CHECK(panel->rect.w == 224.0f);
    CHECK(panel->rect.h == 160.0f);
    CHECK(panel->texWidth == 224);
    CHECK(panel->texHeight == 160);
    CHECK(mat4Transform(panel->world3, Vec3{0, 0, 0}).x == doctest::Approx(64.0f));
    CHECK(mat4Transform(panel->world3, Vec3{0, 0, 0}).y == doctest::Approx(40.0f));

    const std::vector<uint8_t> stacked = rig.restacked();
    const std::vector<uint8_t> flatPx = rig.flat(off.render());
    // A layer is drawn into its own 8-bit target and then stacked: where its antialiased or
    // translucent pixels lie over the layer below, that is one rounding more than the ordinary
    // picture has (the layer's colour and alpha, then the blend), so single edge pixels can be
    // 2 / 255 off; everything else is within 1 / 255.
    const golden::GoldenResult r = golden::compareRgba(stacked.data(), flatPx.data(), kW, kH, 1, 0.0f);
    MESSAGE("coplanar layers stacked vs ordinary: " << r.diffPixels << " of " << kW * kH
                                                    << " px differ by more than 1 / 255, max channel diff "
                                                    << r.maxChannelDiff);
    CHECK(r.maxChannelDiff <= 2u);
    CHECK(r.diffPixels <= 8u);
}

TEST_CASE(
    "wgpu world layers: a still UI draws nothing from its second frame on; one changing layer "
    "redraws alone (T-701)") {
    const std::string css = std::string(kCss) + "#panel { world-layer: on; } #badge { world-layer: on; }";
    Ui ui(css, kBody, /*layers=*/true);
    Rig rig;
    wgpu::WgpuDevice& device = *rig.renderer.device();
    rig.buildAndSubmit(ui.render());
    REQUIRE(rig.layers.layers().size() == 3);
    const uint64_t drawnAfterFirst = rig.layers.totalSubmitted();
    CHECK(drawnAfterFirst == 3u);
    const NodeId panelId = ui.doc->byId("panel").node;
    const NodeId badgeId = ui.doc->byId("badge").node;
    const std::vector<uint8_t> baseBefore = rig.readLayer(*layerOf(rig.layers.layers(), 0));
    const std::vector<uint8_t> badgeBefore = rig.readLayer(*layerOf(rig.layers.layers(), badgeId));
    const std::vector<uint8_t> panelBefore = rig.readLayer(*layerOf(rig.layers.layers(), panelId));

    // Frames 2 and 3 of the still UI: no frame built, no target drawn, nothing sent to the queue.
    for (int frame = 0; frame < 2; ++frame) {
        ui.step();
        const WGPUSubmissionIndex before = device.lastSubmissionIndex();
        rig.buildAndSubmit(ui.render());
        CHECK(rig.layers.stats().layers == 3u);
        CHECK(rig.layers.stats().built == 0u);
        CHECK(rig.layers.stats().drawCommands == 0u);
        CHECK(rig.layers.stats().submitted == 0u);
        CHECK(device.lastSubmissionIndex() == before);
        for (const gpu::WorldLayer& layer : rig.layers.layers()) {
            CHECK(layer.changed == 0);
        }
    }
    CHECK(rig.layers.totalSubmitted() == drawnAfterFirst);

    // Only #chip (inside #panel's layer) changes: that one layer is redrawn, the others' targets
    // are not written.
    CHECK(ui.doc->byId("chip").setStyle("background-color: #40ff80") == Status::kOk);
    ui.step();
    rig.buildAndSubmit(ui.render());
    CHECK(rig.layers.stats().built == 1u);
    CHECK(rig.layers.stats().submitted == 1u);
    CHECK(rig.layers.totalSubmitted() == drawnAfterFirst + 1);
    for (const gpu::WorldLayer& layer : rig.layers.layers()) {
        CHECK(layer.changed == (layer.id == panelId ? 1 : 0));
    }
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), 0)) == baseBefore);
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), badgeId)) == badgeBefore);
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), panelId)) != panelBefore);

    // Moving, tilting and fading a layer changes where the host puts it, not its picture.
    CHECK(ui.doc->byId("badge").setStyle(
              "transform: translate(-20px, 4px) translateZ(60px) rotateY(35deg); opacity: 0.5") ==
          Status::kOk);
    ui.step();
    const WGPUSubmissionIndex before = device.lastSubmissionIndex();
    rig.buildAndSubmit(ui.render());
    CHECK(rig.layers.stats().built == 0u);
    CHECK(device.lastSubmissionIndex() == before);
    const gpu::WorldLayer* badge = layerOf(rig.layers.layers(), badgeId);
    REQUIRE(badge != nullptr);
    CHECK(badge->opacity == doctest::Approx(0.5f));
    CHECK(mat4Transform(badge->world3, Vec3{36, 24, 0}).z == doctest::Approx(60.0f));
    CHECK(rig.readLayer(*badge) == badgeBefore);

    // A layer that goes away gives its target back; the others stay as they are.
    CHECK(ui.doc->byId("badge").setStyle("display: none") == Status::kOk);
    ui.step();
    rig.buildAndSubmit(ui.render());
    CHECK(rig.layers.layers().size() == 2);
    CHECK(rig.layers.stats().built == 0u);
}

TEST_CASE(
    "wgpu world layers: density scales the textures; a device without layer targets is reported "
    "(T-701)") {
    const std::string css = std::string(kCss) + "#panel { world-layer: on; }";
    Ui ui(css, kBody, /*layers=*/true);
    Rig rig;
    rig.layers.setDensity(2.0f);
    rig.buildAndSubmit(ui.render());
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), ui.doc->byId("panel").node);
    REQUIRE(panel != nullptr);
    CHECK(panel->texWidth == 448);
    CHECK(panel->texHeight == 320);
    CHECK(panel->density == 2.0f);
    // The picture is there: the panel's centre (plane point (80, 48)) is its blue.
    const std::vector<uint8_t> px = rig.readLayer(*panel);
    const int x = int((80.0f - panel->rect.x) * 2.0f), y = int((48.0f - panel->rect.y) * 2.0f);
    const uint8_t* p = pixel(px, panel->texWidth, x, y);
    CHECK(int(p[0]) == 0x2f);
    CHECK(int(p[1]) == 0x6f);
    CHECK(int(p[2]) == 0xed);
    CHECK(int(p[3]) == 0xff);
    // A corner of the rectangle is transparent.
    CHECK(int(pixel(px, panel->texWidth, 2, 2)[3]) == 0);

    // The base GpuDevice has no layer targets: build says so and lists nothing.
    struct NoLayers final : gpu::GpuDevice {
        gpu::TextureHandle createTexture(const char*, uint16_t, uint16_t, gpu::TextureFormat,
                                         const void*) override {
            return 1;
        }
        void updateTexture(gpu::TextureHandle, uint16_t, uint16_t, uint16_t, uint16_t, const void*,
                           size_t) override {}
        void destroyTexture(gpu::TextureHandle) override {}
        void submit(const gpu::GpuFrame&) override {}
    } plain;
    gpu::WorldLayerBuilder none;
    none.init(plain);
    CHECK_FALSE(none.build(ui.render(), nullptr, nullptr, nullptr));
    CHECK(std::string(none.lastError()).find("not supported") != std::string::npos);
    CHECK(none.layers().empty());
    // A list painted outside world-layer output has no layers either.
    Ui off(css, kBody, /*layers=*/false);
    CHECK_FALSE(rig.layers.build(off.render(), nullptr, nullptr, nullptr));
}

TEST_CASE("wgpu world layers: every layer's texture matches xt::sw drawing its command range (T-701)") {
    // A tilted layer with a gradient, a rounded border and a drop shadow; a second layer inside it.
    const std::string css = std::string(kCss) +
                            "#panel { world-layer: on; transform: translateZ(80px) rotateY(-25deg); "
                            "background-gradient: linear-gradient(90deg, #2f6fed 0%, #7fe3ff 100%); "
                            "effect: shadow(0px 6px 12px 0px #000000aa); }"
                            "#chip { world-layer: on; transform: translateZ(20px); }"
                            "#badge { world-layer: on; opacity: 0.6; }";
    Ui ui(css, kBody, /*layers=*/true);
    Rig rig;
    const CommandList list = ui.render();
    rig.buildAndSubmit(list);
    REQUIRE(rig.layers.layers().size() == 4);
    REQUIRE(list.worldLayers.size() == 4);
    for (size_t i = 0; i < 4; ++i) {
        const gpu::WorldLayer& layer = rig.layers.layers()[i];
        const WorldLayerRange& range = list.worldLayers[i];
        REQUIRE(layer.id == range.id);
        // The range alone, as a list of its own: logical (0, 0) is the rectangle's corner.
        CommandList part = list;
        part.commands.assign(list.commands.begin() + range.firstCommand,
                             list.commands.begin() + range.firstCommand + range.commandCount);
        part.worldLayers.clear();
        part.viewport = Rect{0, 0, range.rect.w, range.rect.h};
        const std::vector<uint8_t> sw =
            renderSw(part, layer.texWidth, layer.texHeight, Color{0, 0, 0, 0}, nullptr, nullptr);
        // xt::sw's buffer holds straight alpha; the layer's texture is premultiplied.
        std::vector<uint8_t> gpuPx = rig.readLayer(layer);
        for (size_t p = 0; p + 3 < gpuPx.size(); p += 4) {
            const unsigned a = gpuPx[p + 3];
            for (size_t c = 0; c < 3 && a != 0 && a != 255; ++c) {
                gpuPx[p + c] = static_cast<uint8_t>(std::min(255u, (gpuPx[p + c] * 255u + a / 2u) / a));
            }
        }
        const std::string what = "layer " + std::to_string(i) + " (" + std::to_string(layer.texWidth) + "x" +
                                 std::to_string(layer.texHeight) + ")";
        CHECK(matches(gpuPx, sw, layer.texWidth, layer.texHeight, what.c_str()));
    }
    // The tilted layer's picture is the untilted element: drawn flat, the tilt is the host's.
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), ui.doc->byId("panel").node);
    REQUIRE(panel != nullptr);
    CHECK(std::fabs(panel->world3.m[8]) > 0.4f);
    const std::vector<uint8_t> px = rig.readLayer(*panel);
    // Near the left and right edges of the 160 px wide box at mid height: opaque.
    const int y = int(48.0f - panel->rect.y);
    CHECK(int(pixel(px, panel->texWidth, int(8.0f - panel->rect.x), y)[3]) == 255);
    CHECK(int(pixel(px, panel->texWidth, int(152.0f - panel->rect.x), y)[3]) == 255);
}

TEST_CASE(
    "wgpu world layers: a layer that starts using the stencil gets a target with one, drawn right (T-703)") {
    // #chip sticks out of #panel on the left; clipping #panel (overflow: hidden, rounded) with the
    // edge fringe off takes a stencil level (33 §33.13 item 4: the target had none before).
    const std::string css = std::string(kCss) + "#panel { world-layer: on; } #chip { left: -30px; }";
    Ui ui(css, kBody, /*layers=*/true);
    Rig rig;
    rig.layers.setEdgeFringe(false);
    const NodeId panelId = ui.doc->byId("panel").node;
    rig.buildAndSubmit(ui.render());
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), panelId);
    REQUIRE(panel != nullptr);
    CHECK(panel->stencil == 0);
    CHECK(rig.layers.stats().stencilLayers == 0u);
    const auto at = [](const gpu::WorldLayer& layer, const std::vector<uint8_t>& px, float x, float y) {
        return pixel(px, layer.texWidth, int(x - layer.rect.x), int(y - layer.rect.y));
    };
    {
        const std::vector<uint8_t> px = rig.readLayer(*panel);
        CHECK(int(at(*panel, px, -15, 32)[3]) == 255);  // the chip, outside the panel, not clipped
    }

    CHECK(ui.doc->byId("panel").setStyle("overflow: hidden") == Status::kOk);
    ui.step();
    rig.buildAndSubmit(ui.render());
    panel = layerOf(rig.layers.layers(), panelId);
    REQUIRE(panel != nullptr);
    CHECK(panel->stencil == 1);
    CHECK(panel->changed == 1);
    CHECK(layerOf(rig.layers.layers(), 0)->changed == 0);
    CHECK(rig.layers.stats().stencilLayers == 1u);
    CHECK(std::string(rig.renderer.device()->lastError()).empty());
    const std::vector<uint8_t> switched = rig.readLayer(*panel);
    CHECK(int(at(*panel, switched, -15, 32)[3]) == 0);  // clipped away
    const uint8_t* chip = at(*panel, switched, 15, 32);
    CHECK(int(chip[0]) == 0xff);
    CHECK(int(chip[1]) == 0xd3);
    CHECK(int(chip[2]) == 0x4d);
    CHECK(int(chip[3]) == 0xff);

    // The same state drawn by a builder that never had a target without stencil: the same picture.
    gpu::WorldLayerBuilder fresh;
    fresh.init(*rig.renderer.device());
    fresh.setEdgeFringe(false);
    REQUIRE(fresh.build(ui.render(), nullptr, nullptr, nullptr));
    fresh.submit();
    const gpu::WorldLayer* reference = layerOf(fresh.layers(), panelId);
    REQUIRE(reference != nullptr);
    CHECK(reference->stencil == 1);
    CHECK(rig.readLayer(*reference) == switched);
    fresh.shutdown();
}

TEST_CASE("wgpu world layers: a 4x device draws layers without stencil through its 4x attachment (T-703)") {
    const std::string css = std::string(kCss) + "#panel { world-layer: on; }";
    Ui ui(css, kBody, /*layers=*/true);
    wgpu::WgpuRenderer renderer;
    wgpu::WgpuInit in = sharedInit();
    in.sampleCount = 4;
    in.error.clear();
    REQUIRE(renderer.init(in));
    renderer.resize(kW, kH);
    gpu::WorldLayerBuilder layers;
    layers.init(*renderer.device());
    for (int frame = 0; frame < 2; ++frame) {  // the second frame redraws through the kept attachment
        if (frame == 1) {
            CHECK(ui.doc->byId("chip").setStyle("background-color: #40ff80") == Status::kOk);
            ui.step();
        }
        REQUIRE_MESSAGE(layers.build(ui.render(), nullptr, nullptr, nullptr), layers.lastError());
        layers.submit();
        CHECK(std::string(renderer.device()->lastError()).empty());
        const gpu::WorldLayer* panel = layerOf(layers.layers(), ui.doc->byId("panel").node);
        REQUIRE(panel != nullptr);
        CHECK(panel->stencil == 0);
        std::vector<uint8_t> px(size_t(panel->texWidth) * panel->texHeight * 4);
        REQUIRE(renderer.readback(renderer.device()->texture(panel->texture), px));
        // The panel's blue below the chip, and the chip's colour of this frame.
        const uint8_t* blue =
            pixel(px, panel->texWidth, int(80.0f - panel->rect.x), int(70.0f - panel->rect.y));
        CHECK(int(blue[0]) == 0x2f);
        CHECK(int(blue[2]) == 0xed);
        const uint8_t* chip =
            pixel(px, panel->texWidth, int(50.0f - panel->rect.x), int(32.0f - panel->rect.y));
        CHECK(int(chip[1]) == (frame == 0 ? 0xd3 : 0xff));
    }
    layers.shutdown();
}
