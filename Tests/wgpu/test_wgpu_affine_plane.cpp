// T-700 (Docs/design/10-backends.md §10.15): an affine plane inside a 3D space is drawn with the 2D
// paths. At 1x, the content of a CSS `perspective` root's own plane builds the same GpuFrame and the
// same pixels as the same content without the root; a translateZ layer draws like the equivalent 2D
// scale + translate. The same holds for xt::sw.
//
// Every case shapes text with FreeType, so the file is empty in a build without text (XT_GOLDEN_TEXT unset,
// e.g. the WSL fuzz tree), as test_wgpu_capabilities.cpp guards its text includes.

#ifdef XT_GOLDEN_TEXT

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/project3d.h"
#include "xt/text/ft_text_shaper.h"
#include "xt/text/glyph_atlas.h"
#include "xt/text_shaper.h"
#include "xt/wgpu/wgpu_device.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 256, kH = 144;
constexpr float kPerspective = 1000.0f;

/// CSS `perspective: d` with `perspective-origin` at `o` (translate(o) * P(d) * translate(-o)).
Mat4 cssPerspective(float d, Vec2 o) {
    Mat4 p = mat4Identity();
    p.m[14] = -1.0f / d;
    return mat4Mul(mat4Translate(Vec3{o.x, o.y, 0}), mat4Mul(p, mat4Translate(Vec3{-o.x, -o.y, 0})));
}

void pushProjection(CommandList& cl) {
    ProjectionCmd c{};
    c.viewProj = cssPerspective(kPerspective, Vec2{kW * 0.5f, kH * 0.5f});
    c.rootXf = Mat2x3{};
    c.boxRect = Rect{0, 0, kW, kH};
    cl.projections.push_back(c);
    cl.commands.push_back(
        Command{CmdKind::PushProjection, {}, static_cast<uint32_t>(cl.projections.size() - 1)});
}

void pushTransform3D(CommandList& cl, const Mat4& world3) {
    Transform3DCmd c{};
    c.world3 = world3;
    cl.transforms3.push_back(c);
    cl.commands.push_back(
        Command{CmdKind::PushTransform3D, {}, static_cast<uint32_t>(cl.transforms3.size() - 1)});
}

void pop(CommandList& cl, CmdKind kind) { cl.commands.push_back(Command{kind, {}, 0}); }

/// Appends an arc from `a0` to `a1` (radians) of radius `r` around `c` to the open polygon `pts`.
void arc(std::vector<Vec2>& pts, Vec2 c, float r, float a0, float a1, int steps) {
    for (int i = 0; i <= steps; ++i) {
        const float a = a0 + (a1 - a0) * static_cast<float>(i) / static_cast<float>(steps);
        pts.push_back(Vec2{c.x + r * std::cos(a), c.y + r * std::sin(a)});
    }
}

PathRef polygonPath(CommandList& cl, const std::vector<Vec2>& pts) {
    PathRef ref{};
    ref.verbOffset = static_cast<uint32_t>(cl.verbs.size());
    ref.pointOffset = static_cast<uint32_t>(cl.points.size());
    for (size_t i = 0; i < pts.size(); ++i) {
        cl.verbs.push_back(i == 0 ? PathVerb::Move : PathVerb::Line);
        cl.points.push_back(pts[i].x);
        cl.points.push_back(pts[i].y);
    }
    cl.verbs.push_back(PathVerb::Close);
    ref.verbCount = static_cast<uint32_t>(cl.verbs.size()) - ref.verbOffset;
    ref.pointCount = static_cast<uint32_t>(cl.points.size()) - ref.pointOffset;
    return ref;
}

/// A rounded rectangle polygon (eight segments per corner).
std::vector<Vec2> roundedRect(Rect r, float radius) {
    constexpr float kPi = 3.14159265f;
    std::vector<Vec2> pts;
    arc(pts, Vec2{r.x + r.w - radius, r.y + radius}, radius, -kPi * 0.5f, 0.0f, 8);
    arc(pts, Vec2{r.x + r.w - radius, r.y + r.h - radius}, radius, 0.0f, kPi * 0.5f, 8);
    arc(pts, Vec2{r.x + radius, r.y + r.h - radius}, radius, kPi * 0.5f, kPi, 8);
    arc(pts, Vec2{r.x + radius, r.y + radius}, radius, kPi, kPi * 1.5f, 8);
    return pts;
}

void fill(CommandList& cl, PathRef path, Color color, const Mat2x3& xf) {
    FillCmd f{};
    f.xf = xf;
    f.path = path;
    f.paint.kind = PaintKind::Solid;
    f.paint.color = color;
    f.rule = FillRule::NonZero;
    f.antialias = 1;
    cl.fills.push_back(f);
    cl.commands.push_back(Command{CmdKind::FillPath, {}, static_cast<uint32_t>(cl.fills.size() - 1)});
}

/// The sheet's content (Docs/tasks/T-700.md item 4): a concave fill, a 2 px ring, an open ring (a
/// border with two sides 0, like challenge-menu's #holo-arc), text, a rounded clip and an opaque box
/// with an outer shadow -- every transform premultiplied by `m`.
void addContent(CommandList& cl, const Mat2x3& m, text::FtTextShaper& shaper) {
    constexpr float kPi = 3.14159265f;
    // Concave L.
    fill(cl,
         polygonPath(cl, {Vec2{8, 8}, Vec2{48, 8}, Vec2{48, 22}, Vec2{22, 22}, Vec2{22, 48}, Vec2{8, 48}}),
         Color{0.2f, 0.6f, 1.0f, 1.0f}, m);
    // A 2 px ring: an outer and a reversed inner 64-gon.
    std::vector<Vec2> ring;
    arc(ring, Vec2{90, 34}, 26, 0.0f, 2.0f * kPi, 64);
    ring.pop_back();
    PathRef ringPath = polygonPath(cl, ring);
    std::vector<Vec2> hole;
    arc(hole, Vec2{90, 34}, 24, 2.0f * kPi, 0.0f, 64);
    hole.pop_back();
    const PathRef holePath = polygonPath(cl, hole);
    ringPath.verbCount += holePath.verbCount;
    ringPath.pointCount += holePath.pointCount;
    fill(cl, ringPath, Color{0.56f, 1.0f, 0.72f, 0.6f}, m);
    // An open ring: three quarters of a 4 px border, the other quarter 0.
    std::vector<Vec2> open;
    arc(open, Vec2{150, 34}, 28, kPi * 0.5f, kPi * 2.0f, 48);
    arc(open, Vec2{150, 34}, 24, kPi * 2.0f, kPi * 0.5f, 48);
    fill(cl, polygonPath(cl, open), Color{0.78f, 1.0f, 0.86f, 0.87f}, m);
    // Text.
    ShapeParams params;
    static const std::vector<std::string_view> family = {"Noto Sans"};
    params.font = FontRequest{std::span<const std::string_view>(family), 400, false};
    params.size = 16.0f;
    ShapeResult shaped;
    shaper.shape("Affine plane 1x", params, shaped);
    REQUIRE(!shaped.runs.empty());
    GlyphsCmd run{};
    run.xf = m;
    run.font = shaped.runs[0].font;
    run.size = 16.0f;
    run.color = Color{0.9f, 0.95f, 0.9f, 1.0f};
    run.glyphOffset = static_cast<uint32_t>(cl.glyphs.size());
    float pen = 8.0f;
    for (const ShapedGlyph& g : shaped.glyphs) {
        cl.glyphs.push_back(g.glyphId);
        cl.glyphPos.push_back(Vec2{pen + g.xOffset, 84.0f + g.yOffset});
        pen += g.xAdvance;
    }
    run.glyphCount = static_cast<uint32_t>(cl.glyphs.size()) - run.glyphOffset;
    cl.glyphRuns.push_back(run);
    cl.commands.push_back(Command{CmdKind::DrawGlyphs, {}, static_cast<uint32_t>(cl.glyphRuns.size() - 1)});
    // A rounded clip over a larger rectangle.
    ClipPathCmd clip{};
    clip.xf = m;
    clip.path = polygonPath(cl, roundedRect(Rect{140, 72, 56, 40}, 10));
    clip.rule = FillRule::NonZero;
    cl.clipPaths.push_back(clip);
    cl.commands.push_back(Command{CmdKind::PushClipPath, {}, static_cast<uint32_t>(cl.clipPaths.size() - 1)});
    fill(cl, polygonPath(cl, {Vec2{130, 66}, Vec2{206, 66}, Vec2{206, 118}, Vec2{130, 118}}),
         Color{1.0f, 0.5f, 0.2f, 1.0f}, m);
    cl.commands.push_back(Command{CmdKind::PopClip, {}, 0});
    // An opaque box with an outer shadow (DrawBoxShadow, 10 §10.12 item 2).
    BoxShadowCmd bs{};
    bs.xf = m;
    bs.rect = Rect{212, 80, 30, 24};
    for (float& r : bs.radii) {
        r = 4.0f;
    }
    bs.offset = Vec2{2, 3};
    bs.blur = 6.0f;
    bs.color = Color{0, 0, 0, 0.7f};
    cl.boxShadows.push_back(bs);
    cl.commands.push_back(
        Command{CmdKind::DrawBoxShadow, {}, static_cast<uint32_t>(cl.boxShadows.size() - 1)});
    fill(cl, polygonPath(cl, roundedRect(Rect{212, 80, 30, 24}, 4)), Color{0.3f, 0.3f, 0.35f, 1.0f}, m);
}

/// A FrameBuilder on the shared wgpu device with its own glyph atlas: builds one list, dumps the frame
/// and draws it into a readable target.
struct PlaneRig {
    wgpu::WgpuRenderer renderer;
    gpu::FrameBuilder builder;
    gpu::GpuFrame frame;
    std::unique_ptr<text::GlyphAtlas> atlas;
    explicit PlaneRig(text::FtTextShaper& shaper) {
        wgpu::WgpuInit in = sharedInit();
        in.sampleCount = 1;
        in.error.clear();
        REQUIRE(renderer.init(in));
        builder.init(*renderer.device());
        builder.setEdgeFringe(true);
        atlas = std::make_unique<text::GlyphAtlas>(shaper, 512);
        builder.setGlyphAtlas(atlas.get());
    }
    ~PlaneRig() {
        builder.setGlyphAtlas(nullptr);
        builder.shutdown();
    }
    PlaneRig(const PlaneRig&) = delete;
    PlaneRig& operator=(const PlaneRig&) = delete;

    std::vector<uint8_t> draw(const CommandList& cl, TextShaper* shaper, std::string& dump) {
        DrawTarget target(kW, kH);
        renderer.resize(kW, kH);
        builder.resize(kW, kH);
        builder.build(cl, shaper, nullptr, nullptr, frame);
        dump.clear();
        gpu::dumpGpuFrame(frame, dump);
        wgpu::WgpuDevice& device = *renderer.device();
        device.setSurfaceClear(true, WGPUColor{0, 0, 0, 1});
        device.setSurface(target.view);
        device.submit(frame);
        device.setSurface(nullptr);
        std::vector<uint8_t> px(static_cast<size_t>(kW) * kH * 4);
        REQUIRE_MESSAGE(renderer.readback(target.texture, px), renderer.lastError());
        return px;
    }
};

bool loadShaper(text::FtTextShaper& shaper) {
    return shaper.addFontFile(std::string(XT_TEST_FONTS_DIR) + "/NotoSans-Regular.subset.ttf");
}

/// xt::sw with the shaper (renderSw draws no text).
std::vector<uint8_t> renderSwText(const CommandList& cl, TextShaper& shaper) {
    std::vector<uint8_t> buf(static_cast<size_t>(kW) * kH * 4, 0);
    sw::SwRenderer renderer;
    sw::SwTarget target{buf.data(), kW, kH, kW * 4};
    renderer.setTarget(target);
    renderer.clear(Color{0, 0, 0, 1});
    renderer.render(cl, &shaper, nullptr, nullptr);
    return buf;
}

size_t coverageCmds(const gpu::GpuFrame& f) {
    size_t n = 0;
    for (const gpu::DrawCmd& c : f.cmds) n += c.coverage != 0 ? 1 : 0;
    return n;
}

}  // namespace

TEST_CASE("wgpu affine plane: a perspective root's own plane builds and draws as 2D content (T-700)") {
    text::FtTextShaper shaper;
    REQUIRE(loadShaper(shaper));
    CommandList flat = makeList(kW, kH);
    addContent(flat, Mat2x3{}, shaper);
    CommandList space = makeList(kW, kH);
    pushProjection(space);
    addContent(space, Mat2x3{}, shaper);
    pop(space, CmdKind::PopProjection);

    PlaneRig a(shaper), b(shaper);
    std::string dumpFlat, dumpSpace;
    const std::vector<uint8_t> pxFlat = a.draw(flat, &shaper, dumpFlat);
    const std::vector<uint8_t> pxSpace = b.draw(space, &shaper, dumpSpace);
    CHECK(coverageCmds(b.frame) > 0);  // the concave fills and the text are coverage quads at 1x
    CHECK(dumpFlat == dumpSpace);
    const golden::GoldenResult r = golden::compareRgba(pxFlat.data(), pxSpace.data(), kW, kH, 0, 0.0f);
    MESSAGE("identity plane vs 2D: " << r.diffPixels << " px differ, max channel diff " << r.maxChannelDiff);
    CHECK(r.diffPixels == 0u);

    // xt::sw takes the same decision (§10.15 item 3).
    const std::vector<uint8_t> swFlat = renderSwText(flat, shaper);
    const std::vector<uint8_t> swSpace = renderSwText(space, shaper);
    const golden::GoldenResult s = golden::compareRgba(swFlat.data(), swSpace.data(), kW, kH, 0, 0.0f);
    MESSAGE("sw identity plane vs 2D: " << s.diffPixels << " px differ, max channel diff "
                                        << s.maxChannelDiff);
    CHECK(s.diffPixels == 0u);
}

TEST_CASE("wgpu affine plane: a translateZ layer draws like the equivalent 2D scale + translate (T-700)") {
    text::FtTextShaper shaper;
    REQUIRE(loadShaper(shaper));
    // translateZ(60px) under perspective 1000px: w = 1 - 60 / 1000 on the whole plane, so the layer is
    // scaled by 1000 / 940 about the perspective origin.
    const Vec2 o{kW * 0.5f, kH * 0.5f};
    const float k = kPerspective / (kPerspective - 60.0f);
    const Mat2x3 equivalent = mul(translation(o), mul(scaling(Vec2{k, k}), translation(Vec2{-o.x, -o.y})));
    CommandList flat = makeList(kW, kH);
    addContent(flat, equivalent, shaper);
    CommandList space = makeList(kW, kH);
    pushProjection(space);
    pushTransform3D(space, mat4Translate(Vec3{0, 0, 60}));
    addContent(space, Mat2x3{}, shaper);
    pop(space, CmdKind::PopTransform3D);
    pop(space, CmdKind::PopProjection);

    PlaneRig a(shaper), b(shaper);
    std::string dumpFlat, dumpSpace;
    const std::vector<uint8_t> pxFlat = a.draw(flat, &shaper, dumpFlat);
    const std::vector<uint8_t> pxSpace = b.draw(space, &shaper, dumpSpace);
    CHECK(coverageCmds(b.frame) > 0);
    const golden::GoldenResult r = golden::compareRgba(pxFlat.data(), pxSpace.data(), kW, kH, 1, 0.0f);
    MESSAGE("translateZ plane vs 2D scale+translate: "
            << r.diffPixels << " px differ (any channel), max channel diff " << r.maxChannelDiff);
    CHECK(r.maxChannelDiff <= 1u);

    const std::vector<uint8_t> swFlat = renderSwText(flat, shaper);
    const std::vector<uint8_t> swSpace = renderSwText(space, shaper);
    const golden::GoldenResult s = golden::compareRgba(swFlat.data(), swSpace.data(), kW, kH, 1, 0.0f);
    MESSAGE("sw translateZ plane vs 2D: " << s.diffPixels << " px differ, max channel diff "
                                          << s.maxChannelDiff);
    CHECK(s.maxChannelDiff <= 1u);
}

TEST_CASE("wgpu affine plane: a rotateY layer stays projected (T-700)") {
    text::FtTextShaper shaper;
    REQUIRE(loadShaper(shaper));
    CommandList space = makeList(kW, kH);
    pushProjection(space);
    pushTransform3D(space, mat4RotateY(20.0f));
    addContent(space, Mat2x3{}, shaper);
    pop(space, CmdKind::PopTransform3D);
    pop(space, CmdKind::PopProjection);
    PlaneRig rig(shaper);
    std::string dump;
    (void)rig.draw(space, &shaper, dump);
    CHECK(coverageCmds(rig.frame) == 0u);  // still the projected (stencil / outline) paths
}

TEST_CASE(
    "wgpu affine plane: a rotated clip of the root plane around a translateZ layer with its own clip "
    "(T-700)") {
    text::FtTextShaper shaper;
    REQUIRE(loadShaper(shaper));
    const Vec2 o{kW * 0.5f, kH * 0.5f};
    const float k = kPerspective / (kPerspective - 60.0f);
    const Mat2x3 fold = mul(translation(o), mul(scaling(Vec2{k, k}), translation(Vec2{-o.x, -o.y})));
    // The outer clip (root plane) holds a fill and the layer; the layer holds a clip of its own. The
    // clip scan of the outer clip must not bound (or remember) the layer's commands in the root plane.
    auto build = [&](bool space) {
        CommandList cl = makeList(kW, kH);
        if (space) pushProjection(cl);
        const Mat2x3 outer = mul(translation(Vec2{128, 72}), rotationDeg(8.0f));
        ClipRectCmd c{};
        c.xf = outer;
        c.rect = Rect{-100, -56, 200, 112};
        cl.clipRects.push_back(c);
        cl.commands.push_back(
            Command{CmdKind::PushClipRect, {}, static_cast<uint32_t>(cl.clipRects.size() - 1)});
        fill(cl, polygonPath(cl, {Vec2{0, 0}, Vec2{256, 0}, Vec2{256, 144}, Vec2{0, 144}}),
             Color{0.1f, 0.2f, 0.1f, 1}, Mat2x3{});
        if (space) pushTransform3D(cl, mat4Translate(Vec3{0, 0, 60}));
        const Mat2x3 m = space ? Mat2x3{} : fold;
        ClipRectCmd inner{};
        inner.xf = mul(m, mul(translation(Vec2{150, 70}), rotationDeg(-15.0f)));
        inner.rect = Rect{-40, -24, 80, 48};
        cl.clipRects.push_back(inner);
        cl.commands.push_back(
            Command{CmdKind::PushClipRect, {}, static_cast<uint32_t>(cl.clipRects.size() - 1)});
        addContent(cl, m, shaper);
        cl.commands.push_back(Command{CmdKind::PopClip, {}, 0});
        if (space) pop(cl, CmdKind::PopTransform3D);
        cl.commands.push_back(Command{CmdKind::PopClip, {}, 0});
        if (space) pop(cl, CmdKind::PopProjection);
        return cl;
    };
    const CommandList flat = build(false);
    const CommandList space = build(true);
    PlaneRig a(shaper), b(shaper);
    std::string dumpFlat, dumpSpace;
    const std::vector<uint8_t> pxFlat = a.draw(flat, &shaper, dumpFlat);
    const std::vector<uint8_t> pxSpace = b.draw(space, &shaper, dumpSpace);
    MESSAGE("mask layers: 2D " << a.builder.lastClipStats().maskLayers << ", 3D "
                               << b.builder.lastClipStats().maskLayers);
    CHECK(b.builder.lastClipStats().maskLayers == a.builder.lastClipStats().maskLayers);
    const golden::GoldenResult r = golden::compareRgba(pxFlat.data(), pxSpace.data(), kW, kH, 1, 0.0f);
    MESSAGE("clips around a translateZ layer vs 2D: " << r.diffPixels << " px differ, max channel diff "
                                                      << r.maxChannelDiff);
    CHECK(r.maxChannelDiff <= 1u);
}

#endif  // XT_GOLDEN_TEXT
