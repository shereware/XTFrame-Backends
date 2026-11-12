// T-691 (Docs/design/10-backends.md §10.14, Docs/tasks/T-691.md item 5): a device that says it cannot
// draw coverage quads or Shading::BoxShadow -- the real wgpu device behind a wrapper whose two
// capability queries return false and which forwards everything else -- never receives either
// draw, over every golden fixture (scene, Lottie and UI, 1x); box shadows wired through
// gpu::applyDeviceCapabilities are byte-identical to the real device with
// Context::setBoxShadowCommands(false); a DrawBoxShadow handed to the builder anyway is skipped,
// counted and reported once; without coverage, concave shapes are drawn exactly as the real device
// draws them with the edge fringe off (stencil-then-cover, as before T-660b).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "golden_common.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/compile/ir_writer.h"
#include "xt/gpu/frame_builder.h"
#include "xt/ir.h"
#include "xt/node_arena.h"
#include "xt/painter.h"
#include "xt/scene_instance.h"
#include "xt/sw/sw_png.h"
#ifdef XT_GOLDEN_TEXT
#include "xt/text/ft_text_shaper.h"
#include "xt/text/glyph_atlas.h"
#endif

using namespace xt;
using namespace xt::wgpu::test;

namespace {

/// The real WgpuDevice with chosen answers to the two T-691 capability queries; every other call is
/// forwarded. Counts the draws a device without the capability must never receive.
class CapDevice : public gpu::GpuDevice {
public:
    CapDevice(wgpu::WgpuDevice& inner, bool coverage, bool boxShadow)
        : m_inner(inner), m_coverage(coverage), m_boxShadow(boxShadow) {}

    uint32_t coverageDraws = 0;   ///< draws with DrawCmd::coverage received while !coverage
    uint32_t boxShadowDraws = 0;  ///< Shading::BoxShadow draws received while !boxShadow
    uint32_t coverageTextures = 0;

    bool supportsCoverageDraws() const override { return m_coverage; }
    bool supportsBoxShadowShading() const override { return m_boxShadow; }

    gpu::TextureHandle createTexture(const char* debugName, uint16_t w, uint16_t h, gpu::TextureFormat fmt,
                                     const void* initial) override {
        if (debugName != nullptr && std::strcmp(debugName, "xt coverage atlas") == 0) {
            ++coverageTextures;
        }
        return m_inner.createTexture(debugName, w, h, fmt, initial);
    }
    void updateTexture(gpu::TextureHandle tex, uint16_t x, uint16_t y, uint16_t w, uint16_t h,
                       const void* data, size_t rowPitch) override {
        m_inner.updateTexture(tex, x, y, w, h, data, rowPitch);
    }
    void destroyTexture(gpu::TextureHandle tex) override { m_inner.destroyTexture(tex); }
    void submit(const gpu::GpuFrame& frame) override {
        for (const gpu::DrawCmd& c : frame.cmds) {
            if (!m_coverage && c.coverage != 0) {
                ++coverageDraws;
            }
            if (!m_boxShadow && c.shading == gpu::Shading::BoxShadow) {
                ++boxShadowDraws;
            }
        }
        m_inner.submit(frame);
    }
    gpu::TextureHandle acquireCachedTarget(uint16_t w, uint16_t h, gpu::TextureFormat fmt) override {
        return m_inner.acquireCachedTarget(w, h, fmt);
    }
    void releaseCachedTarget(gpu::TextureHandle target) override { m_inner.releaseCachedTarget(target); }
    uint32_t prepareCustomEffect(std::string_view glsl, const char* names, const uint8_t* types,
                                 uint32_t count) override {
        return m_inner.prepareCustomEffect(glsl, names, types, count);
    }
    uint32_t maxTextureSize() const override { return m_inner.maxTextureSize(); }
    bool supportsDirectTarget() const override { return m_inner.supportsDirectTarget(); }
    gpu::TextureHandle acquireBandTarget(uint16_t w, uint16_t h) override {
        return m_inner.acquireBandTarget(w, h);
    }
    void releaseBandTarget(gpu::TextureHandle target) override { m_inner.releaseBandTarget(target); }
    bool sharesImageTextures() const override { return m_inner.sharesImageTextures(); }
    gpu::TextureHandle findSharedImage(const gpu::ImageKey& key, uint16_t w, uint16_t h,
                                       gpu::TextureFormat fmt) override {
        return m_inner.findSharedImage(key, w, h, fmt);
    }
    gpu::TextureHandle createSharedImage(const gpu::ImageKey& key, const char* debugName, uint16_t w,
                                         uint16_t h, gpu::TextureFormat fmt, const void* initial) override {
        return m_inner.createSharedImage(key, debugName, w, h, fmt, initial);
    }
    void releaseSharedImage(gpu::TextureHandle tex, bool keep) override {
        m_inner.releaseSharedImage(tex, keep);
    }
    uint32_t requestMipChain(gpu::TextureHandle tex, const void* storedLevels,
                             uint32_t storedLevelCount) override {
        return m_inner.requestMipChain(tex, storedLevels, storedLevelCount);
    }

private:
    wgpu::WgpuDevice& m_inner;
    bool m_coverage;
    bool m_boxShadow;
};

/// A FrameBuilder over a CapDevice on the shared headless device: draws a list into a fresh target
/// cleared to the fixture's colour and reads it back (as xt_golden --backend wgpu does).
struct CapRig {
    wgpu::WgpuRenderer renderer;  // owns the WgpuDevice and does the readback
    std::unique_ptr<CapDevice> device;
    gpu::FrameBuilder builder;
    gpu::GpuFrame frame;

    CapRig(bool coverage, bool boxShadow, bool edgeFringe = true) {
        wgpu::WgpuInit in = sharedInit();
        in.sampleCount = 1;
        in.error.clear();
        REQUIRE(renderer.init(in));
        device = std::make_unique<CapDevice>(*renderer.device(), coverage, boxShadow);
        builder.init(*device);
        builder.setEdgeFringe(edgeFringe);
    }
    ~CapRig() { builder.shutdown(); }
    CapRig(const CapRig&) = delete;
    CapRig& operator=(const CapRig&) = delete;

    std::vector<uint8_t> draw(int w, int h, Color clear, const CommandList& cmds, TextShaper* shaper,
                              AssetResolver* assets, const IrView* ir) {
        builder.invalidateAllAssets();  // each fixture starts with no image textures (golden_wgpu.cpp)
        DrawTarget target(w, h);
        renderer.resize(w, h);
        builder.resize(w, h);
#ifdef XT_GOLDEN_TEXT
        std::unique_ptr<text::GlyphAtlas> atlas;
        if (shaper != nullptr) {
            atlas = std::make_unique<text::GlyphAtlas>(*static_cast<text::FtTextShaper*>(shaper));
        }
        builder.setGlyphAtlas(atlas.get());
#endif
        builder.build(cmds, shaper, assets, ir, frame);
#ifdef XT_GOLDEN_TEXT
        builder.setGlyphAtlas(nullptr);
#endif
        wgpu::WgpuDevice& inner = *renderer.device();
        inner.setSurfaceClear(true,
                              WGPUColor{clear.r * clear.a, clear.g * clear.a, clear.b * clear.a, clear.a});
        inner.setSurface(target.view);
        device->submit(frame);
        inner.setSurface(nullptr);
        std::vector<uint8_t> px(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
        REQUIRE_MESSAGE(renderer.readback(target.texture, px), renderer.lastError());
        return px;
    }

    golden::UiBackend uiBackend() {
        return [this](const golden::UiFixture& f, const CommandList& cmds, TextShaper* shaper,
                      AssetResolver* assets, const IrView* ir) {
            return draw(f.width, f.height, f.clearColor, cmds, shaper, assets, ir);
        };
    }
    golden::FrameBackend frameBackend() {
        return [this](int w, int h, const Color& clear, const CommandList& cmds, TextShaper* shaper,
                      AssetResolver* assets,
                      const IrView* ir) { return draw(w, h, clear, cmds, shaper, assets, ir); };
    }
    /// The repository's wiring of a Context to this rig's device (T-691 item 3a).
    std::function<void(Context&)> wiring() {
        return [this](Context& ctx) { gpu::applyDeviceCapabilities(*device, ctx); };
    }
};

/// A golden scene fixture painted as xt_golden's renderFixture paints it.
std::vector<uint8_t> drawSceneFixture(const golden::GoldenFixture& fx, CapRig& rig) {
    compile::IrWriter writer;
    fx.build(writer);
    std::vector<uint8_t> bytes = writer.finish();
    Result<IrView> view = IrView::open(IrBlob{bytes.data(), bytes.size()});
    REQUIRE(view.ok());
    NodeArena arena;
    NodeId container = arena.create(NodeKind::Group);
    SceneInstance inst;
    REQUIRE(instantiateScene(view.value(), 0, arena, container, inst) == Status::kOk);
    updateWorldTransforms(arena, inst.root, true);
    Painter painter;
    CommandList cmds;
    painter.paint(arena, inst.root, cmds);
    cmds.viewport = Rect{0, 0, static_cast<float>(fx.width), static_cast<float>(fx.height)};
    cmds.devicePixelRatio = fx.dpr;
    FileAssetResolver resolver(std::string(XT_GOLDEN_DIR), sw::decodeImageBytes);
    const int w = static_cast<int>(static_cast<float>(fx.width) * fx.dpr);
    const int h = static_cast<int>(static_cast<float>(fx.height) * fx.dpr);
    return rig.draw(w, h, fx.clearColor, cmds, nullptr, &resolver, &view.value());
}

size_t countKind(const CommandList& cl, CmdKind kind) {
    size_t n = 0;
    for (const Command& c : cl.commands) {
        n += c.kind == kind ? 1 : 0;
    }
    return n;
}

void addBoxShadow(CommandList& cl, Rect rect, float radius, float blur) {
    BoxShadowCmd bs{};
    bs.xf = Mat2x3{};
    bs.rect = rect;
    for (float& r : bs.radii) {
        r = radius;
    }
    bs.offset = Vec2{3, 4};
    bs.blur = blur;
    bs.color = Color{0, 0, 0, 0.6f};
    cl.commands.push_back(Command{CmdKind::DrawBoxShadow, {}, static_cast<uint32_t>(cl.boxShadows.size())});
    cl.boxShadows.push_back(bs);
}

}  // namespace

TEST_CASE("wgpu capabilities: a device without both draws none of them over every golden fixture (T-691)") {
    CapRig rig(false, false);
    uint64_t coverageFallbacks = 0;
    int drawn = 0;
    auto after = [&](const std::string& name, const std::vector<uint8_t>& px) {
        CAPTURE(name);
        CHECK(!px.empty());
        CHECK(rig.builder.lastBuildStats().boxShadowSkipped == 0u);  // the wiring kept them effects
        coverageFallbacks += rig.builder.lastBuildStats().coverageFallbacks;
        ++drawn;
    };
    for (const golden::GoldenFixture& fx : golden::fixtures()) {
        after(fx.name, drawSceneFixture(fx, rig));
    }
    const int scenes = drawn;
    const std::string lottieDir = XT_GOLDEN_LOTTIE_DIR;
    for (const golden::LottieFixture& fx : golden::loadLottieManifest(lottieDir + "/manifest.txt")) {
        after(fx.name, golden::renderLottieFixture(fx, lottieDir, rig.frameBackend()));
    }
    const int lotties = drawn - scenes;
    const std::string uiDir = XT_GOLDEN_UI_DIR;
    for (const golden::UiFixture& fx : golden::loadUiManifest(uiDir + "/manifest.txt")) {
        after(fx.name, golden::renderUiFixture(fx, uiDir, rig.uiBackend(), rig.wiring()));
    }
    CHECK(golden::manifestErrors() == 0);
    MESSAGE(drawn << " fixtures (" << scenes << " scene, " << lotties << " Lottie, "
                  << drawn - scenes - lotties << " UI), " << coverageFallbacks
                  << " coverage shapes drawn the earlier way");
    CHECK(scenes >= 10);
    CHECK(lotties > 0);
    CHECK(drawn > 150);
    CHECK(rig.device->coverageDraws == 0u);
    CHECK(rig.device->boxShadowDraws == 0u);
    CHECK(rig.device->coverageTextures == 0u);
    CHECK(coverageFallbacks > 0u);  // the fixtures do exercise the coverage paths
}

TEST_CASE(
    "wgpu capabilities: wired box shadows equal the real device with setBoxShadowCommands(false) (T-691)") {
    // The coverage capability stays true here so the two runs differ only in the box shadows.
    const char* names[] = {"ui_fx_shadow", "ui_fx_shadow_inset", "ui_box_shadow", "ui_box_shadow_kept",
                           "ui_box_shadow_relaxed"};
    const std::string uiDir = XT_GOLDEN_UI_DIR;
    const std::vector<golden::UiFixture> all = golden::loadUiManifest(uiDir + "/manifest.txt");
    CapRig real(true, true);
    CapRig noShadow(true, false);
    size_t converted = 0;
    uint32_t skippedUnwired = 0;
    bool unwiredDiffers = false;
    for (const char* name : names) {
        CAPTURE(name);
        const golden::UiFixture* fx = nullptr;
        for (const golden::UiFixture& f : all) {
            fx = f.name == name ? &f : fx;
        }
        REQUIRE(fx != nullptr);
        // How many shadows the default (converting) paint makes of this fixture.
        (void)golden::renderUiFixture(*fx, uiDir,
                                      [&](const golden::UiFixture&, const CommandList& cmds, TextShaper*,
                                          AssetResolver*, const IrView*) {
                                          converted += countKind(cmds, CmdKind::DrawBoxShadow);
                                          return std::vector<uint8_t>{};
                                      });
        const std::vector<uint8_t> expected = golden::renderUiFixture(
            *fx, uiDir, real.uiBackend(), [](Context& ctx) { ctx.setBoxShadowCommands(false); });
        const std::vector<uint8_t> wired =
            golden::renderUiFixture(*fx, uiDir, noShadow.uiBackend(), noShadow.wiring());
        REQUIRE(!expected.empty());
        CHECK(wired == expected);
        CHECK(noShadow.builder.lastBuildStats().boxShadowSkipped == 0u);
        // Not wired: the converted shadows reach the builder and are left out.
        const std::vector<uint8_t> unwired = golden::renderUiFixture(*fx, uiDir, noShadow.uiBackend());
        skippedUnwired += noShadow.builder.lastBuildStats().boxShadowSkipped;
        unwiredDiffers = unwiredDiffers || unwired != expected;
    }
    MESSAGE(converted << " DrawBoxShadow commands in the converting paint of these fixtures");
    CHECK(converted > 0u);
    CHECK(skippedUnwired > 0u);
    CHECK(unwiredDiffers);
    CHECK(noShadow.device->boxShadowDraws == 0u);
}

TEST_CASE(
    "wgpu capabilities: a DrawBoxShadow reaching the builder is skipped, counted, reported once (T-691)") {
    constexpr int kW = 160, kH = 100;
    CommandList cl = makeList(kW, kH);
    addFillRect(cl, 0, 0, kW, kH, Color{1, 1, 1, 1});
    addBoxShadow(cl, Rect{20, 20, 50, 40}, 8, 12);    // blurred: needs Shading::BoxShadow
    addBoxShadow(cl, Rect{90, 20, 50, 40}, 8, 0.0f);  // sharp: drawn as a path either way
    CapRig rig(false, false);
    const std::vector<uint8_t> first = rig.draw(kW, kH, Color{1, 1, 1, 1}, cl, nullptr, nullptr, nullptr);
    CHECK(rig.device->boxShadowDraws == 0u);
    CHECK(rig.builder.lastBuildStats().boxShadowSkipped == 1u);
    const std::string message = rig.builder.lastError();
    CHECK(message.find("Context::setBoxShadowCommands(false)") != std::string::npos);
    (void)rig.draw(kW, kH, Color{1, 1, 1, 1}, cl, nullptr, nullptr, nullptr);
    CHECK(rig.builder.lastBuildStats().boxShadowSkipped == 1u);
    CHECK(std::string(rig.builder.lastError()).empty());  // once
    // The sharp one is drawn (its pixels darker than the background), the blurred one is not.
    CHECK(pixel(first, kW, 115, 45)[0] < 255);
    CHECK(pixel(first, kW, 45, 65)[0] == 255);
}

TEST_CASE(
    "wgpu capabilities: without coverage, concave shapes match the real device with the fringe off (T-691)") {
    // Concave fills, a concave stroke and a concave clip path: before T-660b (and with the fringe off
    // today) all stencil-then-cover. A device without coverage draws must get exactly that; the real
    // device at 1x (fringe on) draws them from the coverage atlas instead.
    constexpr int kW = 200, kH = 160;
    CommandList cl = makeList(kW, kH);
    addFill(
        cl,
        addPolygon(cl, {Vec2{20, 20}, Vec2{80, 25}, Vec2{50, 45}, Vec2{85, 75}, Vec2{25, 70}, Vec2{40, 45}}),
        Color{0.8f, 0.2f, 0.1f, 1});
    // Two overlapping convex contours, even-odd: neither a ring nor disjoint pieces.
    PathRef overlap = addPolygon(cl, {Vec2{110, 15}, Vec2{160, 18}, Vec2{156, 62}, Vec2{106, 58}});
    appendNgon(cl, overlap, Vec2{160, 55}, 24, 7);
    addFill(cl, overlap, Color{0.1f, 0.3f, 0.7f, 1}, FillRule::EvenOdd,
            mul(translation(Vec2{3, 2}), rotationDeg(4)));
    {
        StrokeCmd s{};
        s.xf = Mat2x3{};
        s.path = emptyPath(cl);
        const Vec2 pts[] = {Vec2{20, 100}, Vec2{60, 140}, Vec2{75, 95}, Vec2{110, 145}};
        for (int i = 0; i < 4; ++i) {
            cl.verbs.push_back(i == 0 ? PathVerb::Move : PathVerb::Line);
            cl.points.push_back(pts[i].x);
            cl.points.push_back(pts[i].y);
        }
        s.path.verbCount = static_cast<uint32_t>(cl.verbs.size()) - s.path.verbOffset;
        s.path.pointCount = static_cast<uint32_t>(cl.points.size()) - s.path.pointOffset;
        s.paint.kind = PaintKind::Solid;
        s.paint.color = Color{0.1f, 0.6f, 0.2f, 1};
        s.width = 7;
        s.cap = LineCap::Butt;
        s.join = LineJoin::Miter;
        s.miterLimit = 4;
        cl.strokes.push_back(s);
        cl.commands.push_back(Command{CmdKind::StrokePath, {}, static_cast<uint32_t>(cl.strokes.size() - 1)});
    }
    {
        ClipPathCmd c{};
        c.xf = Mat2x3{};
        c.path =
            addPolygon(cl, {Vec2{125, 95}, Vec2{190, 100}, Vec2{160, 120}, Vec2{185, 150}, Vec2{120, 145}});
        c.rule = FillRule::NonZero;
        cl.clipPaths.push_back(c);
        cl.commands.push_back(
            Command{CmdKind::PushClipPath, {}, static_cast<uint32_t>(cl.clipPaths.size() - 1)});
        addFillRect(cl, 110, 90, 85, 65, Color{0.5f, 0.1f, 0.6f, 1});
        addPopClip(cl);
    }
    const Color white{1, 1, 1, 1};
    CapRig withoutCoverage(false, true);
    const std::vector<uint8_t> got = withoutCoverage.draw(kW, kH, white, cl, nullptr, nullptr, nullptr);
    const uint32_t fallbacks = withoutCoverage.builder.lastBuildStats().coverageFallbacks;
    CapRig fringeOff(true, true, false);
    const std::vector<uint8_t> expected = fringeOff.draw(kW, kH, white, cl, nullptr, nullptr, nullptr);
    CapRig real(true, true);
    const std::vector<uint8_t> today = real.draw(kW, kH, white, cl, nullptr, nullptr, nullptr);
    MESSAGE(fallbacks << " coverage shapes drawn the earlier way");
    CHECK(withoutCoverage.device->coverageDraws == 0u);
    CHECK(withoutCoverage.device->coverageTextures == 0u);
    CHECK(fallbacks == 4u);
    CHECK(got == expected);
    CHECK(real.builder.lastCoverageStats().misses > 0u);  // the real device does use coverage here
    CHECK(today != expected);
}
