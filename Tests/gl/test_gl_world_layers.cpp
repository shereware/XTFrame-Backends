#include <SDL.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "golden_common.h"
#include "xt/command_list.h"
#include "xt/compile/compile.h"
#include "xt/context.h"
#include "xt/document.h"
#include "xt/element_ref.h"
#include "xt/gl/gl_renderer.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/gpu/world_layers.h"
#include "xt/text_shaper.h"

/// T-701 (Docs/design/33-world-layers.md §33.4, §33.8; Docs/tasks/T-701.md): world layers on the GL
/// device -- the layer targets, `submitLayer` (drawn into the device's scratch and blitted, rows
/// top-down), unchanged layers left alone, and the layers stacked back equal to the ordinary picture.

using namespace xt;

namespace {

constexpr int kW = 320, kH = 200;

void* sdlLoadProcLayers(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

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

/// A GlRenderer drawing into its own single-sample FBO, and a layer builder on its device.
struct Rig {
    gl::GLuint fbo = 0, color = 0, rbo = 0;
    gl::GlRenderer renderer;
    gpu::WorldLayerBuilder layers;
    gpu::GpuFrame quads;

    Rig() {
        REQUIRE(renderer.init(sdlLoadProcLayers));
        using namespace xt::gl;
        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8), kW, kH, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, kW, kH);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
        REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        renderer.resize(kW, kH);
        REQUIRE(renderer.device() != nullptr);
        layers.init(*renderer.device());
    }
    ~Rig() {
        layers.shutdown();
        renderer.shutdown();
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteRenderbuffers(1, &rbo);
        glDeleteTextures(1, &color);
    }
    Rig(const Rig&) = delete;
    Rig& operator=(const Rig&) = delete;

    void bindAndClear() {
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, kW, kH);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilMask(0xFFu);
        glClearColor(0, 0, 0, 0);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
    }
    /// The FBO as premultiplied RGBA8, row 0 = top.
    std::vector<uint8_t> read() {
        using namespace xt::gl;
        std::vector<uint8_t> flipped(size_t(kW) * kH * 4), out(flipped.size());
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, kW, kH, GL_RGBA, GL_UNSIGNED_BYTE, flipped.data());
        const size_t row = size_t(kW) * 4;
        for (int y = 0; y < kH; ++y) {
            std::memcpy(out.data() + size_t(y) * row, flipped.data() + size_t(kH - 1 - y) * row, row);
        }
        return out;
    }
    std::vector<uint8_t> flat(const CommandList& list) {
        bindAndClear();
        renderer.render(list, nullptr, nullptr);
        CHECK(std::string(renderer.lastError()).empty());
        return read();
    }
    void buildAndSubmit(const CommandList& list) {
        REQUIRE_MESSAGE(layers.build(list, nullptr, nullptr, nullptr), layers.lastError());
        layers.submit();
    }
    std::vector<uint8_t> readLayer(const gpu::WorldLayer& layer) {
        std::vector<uint8_t> px;
        int w = 0, h = 0;
        REQUIRE(renderer.readLayerTarget(layer.texture, px, w, h));
        REQUIRE(w == layer.texWidth);
        REQUIRE(h == layer.texHeight);
        return px;
    }
    std::vector<uint8_t> restacked() {
        gpu::buildWorldLayerQuads(layers.layers(), mat4Identity(), kW, kH, /*clear=*/false, quads);
        bindAndClear();
        renderer.submitFrame(quads);
        CHECK(std::string(renderer.lastError()).empty());
        return read();
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

TEST_CASE("gl world layers: one layer equals the ordinary picture; coplanar layers stack back (T-701)") {
    Rig rig;
    {
        Ui on(kCss, kBody, /*layers=*/true);
        Ui off(kCss, kBody, /*layers=*/false);
        rig.buildAndSubmit(on.render());
        REQUIRE(rig.layers.layers().size() == 1);
        const std::vector<uint8_t> layerPx = rig.readLayer(rig.layers.layers()[0]);
        const std::vector<uint8_t> flatPx = rig.flat(off.render());
        const golden::GoldenResult r = golden::compareRgba(layerPx.data(), flatPx.data(), kW, kH, 0, 0.0f);
        MESSAGE("gl one layer vs ordinary: " << r.diffPixels << " px differ, max channel diff "
                                             << r.maxChannelDiff);
        CHECK(r.diffPixels == 0u);
    }
    const std::string css = std::string(kCss) + "#panel { world-layer: on; } #badge { world-layer: on; }";
    Ui on(css, kBody, /*layers=*/true);
    Ui off(css, kBody, /*layers=*/false);
    rig.buildAndSubmit(on.render());
    REQUIRE(rig.layers.layers().size() == 3);
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), on.doc->byId("panel").node);
    REQUIRE(panel != nullptr);
    CHECK(panel->texWidth == 224);
    CHECK(panel->texHeight == 160);
    // Rows are top-down: the chip (yellow) sits in the upper half of the panel's texture.
    const std::vector<uint8_t> panelPx = rig.readLayer(*panel);
    const size_t at = (size_t(32 + 32) * panel->texWidth + size_t(32 + 50)) * 4;
    CHECK(int(panelPx[at]) == 0xff);
    CHECK(int(panelPx[at + 1]) == 0xd3);
    CHECK(int(panelPx[at + 2]) == 0x4d);

    const std::vector<uint8_t> stacked = rig.restacked();
    const std::vector<uint8_t> flatPx = rig.flat(off.render());
    const golden::GoldenResult r = golden::compareRgba(stacked.data(), flatPx.data(), kW, kH, 1, 0.0f);
    MESSAGE("gl coplanar layers stacked vs ordinary: " << r.diffPixels << " of " << kW * kH
                                                       << " px differ by more than 1 / 255, max channel diff "
                                                       << r.maxChannelDiff);
    CHECK(r.maxChannelDiff <= 2u);
    CHECK(r.diffPixels <= 8u);
}

TEST_CASE(
    "gl world layers: a still UI draws nothing from its second frame on; one changing layer "
    "redraws alone (T-701)") {
    const std::string css = std::string(kCss) + "#panel { world-layer: on; } #badge { world-layer: on; }";
    Ui ui(css, kBody, /*layers=*/true);
    Rig rig;
    rig.buildAndSubmit(ui.render());
    REQUIRE(rig.layers.layers().size() == 3);
    CHECK(rig.layers.totalSubmitted() == 3u);
    const NodeId panelId = ui.doc->byId("panel").node;
    const NodeId badgeId = ui.doc->byId("badge").node;
    const std::vector<uint8_t> baseBefore = rig.readLayer(*layerOf(rig.layers.layers(), 0));
    const std::vector<uint8_t> badgeBefore = rig.readLayer(*layerOf(rig.layers.layers(), badgeId));

    for (int frame = 0; frame < 2; ++frame) {
        ui.step();
        rig.buildAndSubmit(ui.render());
        CHECK(rig.layers.stats().built == 0u);
        CHECK(rig.layers.stats().submitted == 0u);
    }
    CHECK(rig.layers.totalSubmitted() == 3u);

    CHECK(ui.doc->byId("chip").setStyle("background-color: #40ff80") == Status::kOk);
    ui.step();
    rig.buildAndSubmit(ui.render());
    CHECK(rig.layers.stats().built == 1u);
    CHECK(rig.layers.totalSubmitted() == 4u);
    for (const gpu::WorldLayer& layer : rig.layers.layers()) {
        CHECK(layer.changed == (layer.id == panelId ? 1 : 0));
    }
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), 0)) == baseBefore);
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), badgeId)) == badgeBefore);
    // A second, larger layer after a smaller one: the scratch grows, both pictures stay right.
    CHECK(ui.doc->byId("panel").setStyle("width: 240px; height: 140px") == Status::kOk);
    ui.step();
    rig.buildAndSubmit(ui.render());
    const gpu::WorldLayer* panel = layerOf(rig.layers.layers(), panelId);
    REQUIRE(panel != nullptr);
    CHECK(panel->texWidth == 288);
    CHECK(panel->changed == 1);
    CHECK(rig.readLayer(*layerOf(rig.layers.layers(), badgeId)) == badgeBefore);
}

TEST_CASE(
    "gl world layers: a layer that starts using the stencil gets a target with one, drawn right (T-703)") {
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
        return &px[(size_t(int(y - layer.rect.y)) * layer.texWidth + size_t(int(x - layer.rect.x))) * 4];
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
    CHECK(std::string(rig.renderer.lastError()).empty());
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
