// T-719 (Docs/design/10-backends.md §10.16 W1, Docs/tasks/T-719.md "測試"): rounded clips drawn as
// analytic clip levels by xt_gl at 1x, against xt::sw (15 §15.3's tolerance: channel 16, 3 % of the
// pixels). The cases of Tests/wgpu/test_wgpu_analytic_clip.cpp (T-718 / T-720) with the same command
// lists -- solid, gradient, coverage and box-shadow leaves, images, text, transformed and nested
// clips, groups whose results are not clipped -- plus the leaf programs those do not reach: the
// outlined box shadow, the trilinear image, straight-alpha (BC7) images and the two particle programs.
// XT_ANALYTIC_CLIP_DUMP=<dir> writes every case's pixels to <dir>/gl_<case>.rgba (raw RGBA8, rows
// top-down), as the wgpu file does with wgpu_<case>.rgba, for the GL / wgpu comparison.

#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "golden_common.h"
#include "rounded_rect.h"  // Code/Runtime/core/src: appendRoundedRectPath
#include "xt/asset_resolver.h"
#include "xt/command_list.h"
#include "xt/compile/compile.h"
#include "xt/document.h"
#include "xt/gl/gl_renderer.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/ir.h"
#include "xt/sw/sw_png.h"
#include "xt/sw/sw_renderer.h"
#include "xt/text_shaper.h"
#include "xt/types.h"
#ifndef XT_GL_NO_TEXT
#include "xt/text/ft_text_shaper.h"
#include "xt/text/glyph_atlas.h"
#endif

using namespace xt;

namespace {

constexpr int kW = 256, kH = 160;
constexpr Color kClear{0.1f, 0.1f, 0.12f, 1};

void* sdlLoadProcClip(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

/// A GlRenderer drawing into its own single-sample RGBA8 + stencil FBO of `w` x `h`.
struct Session {
    gl::GLuint fbo = 0, color = 0, rbo = 0;
    int w = 0, h = 0;
    gl::GlRenderer renderer;

    Session(int width, int height) : w(width), h(height) {
        REQUIRE(renderer.init(sdlLoadProcClip));
        using namespace xt::gl;
        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8), w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_NEAREST));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_NEAREST));
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
        REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        renderer.resize(w, h);
    }
    ~Session() {
        renderer.shutdown();
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteRenderbuffers(1, &rbo);
        glDeleteTextures(1, &color);
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    /// Clears to `clear` (opaque), draws `cl`, returns the pixels with row 0 = top.
    std::vector<uint8_t> render(const CommandList& cl, Color clear, TextShaper* shaper = nullptr,
                                AssetResolver* assets = nullptr, const IrView* ir = nullptr) {
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, w, h);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilMask(0xFFu);
        glClearColor(clear.r, clear.g, clear.b, clear.a);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        renderer.render(cl, shaper, assets, ir);
        CHECK_MESSAGE(std::string(renderer.lastError()).empty(), renderer.lastError());
        std::vector<uint8_t> flipped(static_cast<size_t>(w) * static_cast<size_t>(h) * 4),
            out(flipped.size());
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, flipped.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        const size_t row = static_cast<size_t>(w) * 4;
        for (int y = 0; y < h; ++y) {
            std::copy(flipped.begin() + static_cast<long>(static_cast<size_t>(h - 1 - y) * row),
                      flipped.begin() + static_cast<long>(static_cast<size_t>(h - y) * row),
                      out.begin() + static_cast<long>(static_cast<size_t>(y) * row));
        }
        return out;
    }

    /// What a frame builder on this renderer's device makes of `cl` (the renderer keeps its own
    /// builder's statistics to itself): analytic clip levels, mask layers, fallbacks.
    struct ClipCounts {
        uint32_t analytic = 0, maskLayers = 0, fallbacks = 0;
    };
    ClipCounts clipCounts(const CommandList& cl, TextShaper* shaper = nullptr,
                          AssetResolver* assets = nullptr, const IrView* ir = nullptr) {
        ClipCounts counts;
        gpu::GpuDevice* device = renderer.device();
        REQUIRE(device != nullptr);
        gpu::FrameBuilder builder;
        builder.init(*device);
        builder.resize(w, h);
        builder.setEdgeFringe(true);
        gpu::GpuFrame frame;
        builder.build(cl, shaper, assets, ir, frame);
        counts.analytic = builder.lastBuildStats().analyticClips;
        counts.fallbacks = builder.lastBuildStats().analyticClipFallbacks;
        counts.maskLayers = builder.lastClipStats().maskLayers;
        builder.shutdown();
        return counts;
    }
};

std::vector<uint8_t> renderSw(const CommandList& cl, int w, int h, Color clear, TextShaper* shaper = nullptr,
                              AssetResolver* assets = nullptr, const IrView* ir = nullptr) {
    std::vector<uint8_t> buf(static_cast<size_t>(w) * static_cast<size_t>(h) * 4, 0);
    sw::SwRenderer renderer;
    renderer.setTarget(sw::SwTarget{buf.data(), w, h, w * 4});
    renderer.clear(clear);
    renderer.render(cl, shaper, assets, ir);
    return buf;
}

void dump(const std::vector<uint8_t>& px, const char* slug) {
    std::string dir;
#if defined(_WIN32)
    char* value = nullptr;
    size_t length = 0;
    if (_dupenv_s(&value, &length, "XT_ANALYTIC_CLIP_DUMP") == 0 && value != nullptr) {
        dir = value;
        std::free(value);
    }
#else
    if (const char* value = std::getenv("XT_ANALYTIC_CLIP_DUMP")) {
        dir = value;
    }
#endif
    if (dir.empty()) {
        return;
    }
    std::ofstream out(dir + "/gl_" + slug + ".rgba", std::ios::binary);
    out.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

/// 15 §15.3's GPU/SW tolerance; prints the numbers for the report.
bool matches(const std::vector<uint8_t>& gl, const std::vector<uint8_t>& sw, int w, int h, const char* what,
             float ratio = 0.03f) {
    golden::GoldenResult r = golden::compareRgba(gl.data(), sw.data(), w, h, 16, ratio);
    std::printf("gl vs sw %s: %u px differ, max channel diff %u\n", what, r.diffPixels, r.maxChannelDiff);
    return r.pass;
}

const uint8_t* pixel(const std::vector<uint8_t>& rgba, int w, int x, int y) {
    return rgba.data() + (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
}

// -------------------------------------------------------------------------------------------
// Command list builders, as Tests/wgpu/test_wgpu_draw_util.h and test_wgpu_analytic_clip.cpp.
// -------------------------------------------------------------------------------------------

CommandList makeList(int w, int h) {
    CommandList cl;
    cl.viewport = Rect{0, 0, static_cast<float>(w), static_cast<float>(h)};
    cl.devicePixelRatio = 1;
    return cl;
}

PathRef addPolygon(CommandList& cl, std::initializer_list<Vec2> pts) {
    PathRef ref{};
    ref.verbOffset = static_cast<uint32_t>(cl.verbs.size());
    ref.pointOffset = static_cast<uint32_t>(cl.points.size());
    bool first = true;
    for (Vec2 p : pts) {
        cl.verbs.push_back(first ? PathVerb::Move : PathVerb::Line);
        cl.points.push_back(p.x);
        cl.points.push_back(p.y);
        first = false;
    }
    cl.verbs.push_back(PathVerb::Close);
    ref.verbCount = static_cast<uint32_t>(cl.verbs.size()) - ref.verbOffset;
    ref.pointCount = static_cast<uint32_t>(cl.points.size()) - ref.pointOffset;
    return ref;
}

void addFill(CommandList& cl, PathRef path, Color color) {
    FillCmd f{};
    f.path = path;
    f.paint.kind = PaintKind::Solid;
    f.paint.color = color;
    f.rule = FillRule::NonZero;
    f.antialias = 1;
    cl.fills.push_back(f);
    cl.commands.push_back(Command{CmdKind::FillPath, {}, static_cast<uint32_t>(cl.fills.size() - 1)});
}

PathRef rectPath(CommandList& cl, Rect r) {
    return addPolygon(
        cl, {Vec2{r.x, r.y}, Vec2{r.x + r.w, r.y}, Vec2{r.x + r.w, r.y + r.h}, Vec2{r.x, r.y + r.h}});
}

void addFillRect(CommandList& cl, float x, float y, float w, float h, Color color) {
    addFill(cl, rectPath(cl, Rect{x, y, w, h}), color);
}

/// The PushClipPath of a circular-cornered rectangle, appended, or inserted at `at` (a command index).
void pushRoundedClip(CommandList& cl, Rect box, float tl, float tr, float br, float bl, Mat2x3 xf = Mat2x3{},
                     int at = -1) {
    ClipPathCmd c{};
    c.xf = xf;
    c.path.verbOffset = static_cast<uint32_t>(cl.verbs.size());
    c.path.pointOffset = static_cast<uint32_t>(cl.points.size());
    appendRoundedRectPath(box, tl, tr, br, bl, cl.verbs, cl.points);
    c.path.verbCount = static_cast<uint32_t>(cl.verbs.size()) - c.path.verbOffset;
    c.path.pointCount = static_cast<uint32_t>(cl.points.size()) - c.path.pointOffset;
    c.rule = FillRule::NonZero;
    cl.clipPaths.push_back(c);
    const Command cmd{CmdKind::PushClipPath, {}, static_cast<uint32_t>(cl.clipPaths.size() - 1)};
    if (at < 0) {
        cl.commands.push_back(cmd);
    } else {
        cl.commands.insert(cl.commands.begin() + at, cmd);
    }
}

void pushRoundedClip(CommandList& cl, Rect box, float r, Mat2x3 xf = Mat2x3{}) {
    pushRoundedClip(cl, box, r, r, r, r, xf);
}

void addPopClip(CommandList& cl) { cl.commands.push_back(Command{CmdKind::PopClip, {}, 0}); }

void fillGradient(CommandList& cl, Rect r, Color a, Color b, Mat2x3 xf = Mat2x3{}) {
    FillCmd f{};
    f.xf = xf;
    f.path = rectPath(cl, r);
    f.paint.kind = PaintKind::Linear;
    f.paint.color = Color{1, 1, 1, 1};
    f.paint.p0 = Vec2{r.x, r.y};
    f.paint.p1 = Vec2{r.x + r.w, r.y + r.h};
    f.paint.stopOffset = static_cast<uint32_t>(cl.stops.size());
    cl.stops.push_back(GradientStop{0.0f, a});
    cl.stops.push_back(GradientStop{1.0f, b});
    f.paint.stopCount = 2;
    f.rule = FillRule::NonZero;
    f.antialias = 1;
    cl.fills.push_back(f);
    cl.commands.push_back(Command{CmdKind::FillPath, {}, static_cast<uint32_t>(cl.fills.size() - 1)});
}

ImageData patterned(int w, int h) {
    ImageData img;
    img.width = w;
    img.height = h;
    img.rgba.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t* p = img.rgba.data() +
                         (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4;
            p[0] = static_cast<uint8_t>(64 + (x * 3) % 192);
            p[1] = static_cast<uint8_t>(64 + (y * 5) % 192);
            p[2] = static_cast<uint8_t>(((x / 8 + y / 8) % 2) * 255);
            p[3] = 255;
        }
    }
    return img;
}

class ImageResolver : public AssetResolver {
public:
    explicit ImageResolver(ImageData image) : m_image(std::move(image)) {}
    bool readBytes(std::string_view, std::vector<uint8_t>&) override { return false; }
    bool decodeImage(AssetId asset, const IrView&, ImageData& out) override {
        if (asset != 0) return false;
        out = m_image;
        return true;
    }
    Vec2 imageSize(AssetId, const IrView&) override {
        return Vec2{static_cast<float>(m_image.width), static_cast<float>(m_image.height)};
    }
    const ImageData* imageView(AssetId asset, const IrView&) override {
        return asset == 0 ? &m_image : nullptr;
    }

private:
    ImageData m_image;
};

void addImage(CommandList& cl, Rect dst, Mat2x3 xf = Mat2x3{}) {
    ImageCmd c{};
    c.xf = xf;
    c.asset = 0;
    c.dst = dst;
    c.srcUv = Rect{0, 0, 1, 1};
    c.opacity = 1;
    c.smooth = 1;
    cl.images.push_back(c);
    cl.commands.push_back(Command{CmdKind::DrawImage, {}, static_cast<uint32_t>(cl.images.size() - 1)});
}

BoxShadowCmd& addBoxShadow(CommandList& cl, Rect r, float radius, Vec2 offset, float blur, Color color) {
    BoxShadowCmd bs{};
    bs.rect = r;
    for (float& v : bs.radii) {
        v = radius;
    }
    bs.offset = offset;
    bs.blur = blur;
    bs.color = color;
    cl.boxShadows.push_back(bs);
    cl.commands.push_back(
        Command{CmdKind::DrawBoxShadow, {}, static_cast<uint32_t>(cl.boxShadows.size() - 1)});
    return cl.boxShadows.back();
}

void addPushLayer(CommandList& cl, float opacity, BlendMode blend, Rect hint) {
    LayerCmd l{};
    l.opacity = opacity;
    l.blend = blend;
    l.boundsHint = hint;
    cl.layers.push_back(l);
    cl.commands.push_back(Command{CmdKind::PushLayer, {}, static_cast<uint32_t>(cl.layers.size() - 1)});
}

void addPopLayer(CommandList& cl) { cl.commands.push_back(Command{CmdKind::PopLayer, {}, 0}); }

void addPushEffect(CommandList& cl, EffectInstance effect, Rect boundsHint, Rect contentBounds) {
    EffectCmd c{};
    c.effectOffset = static_cast<uint32_t>(cl.effects.size());
    c.effectCount = 1;
    cl.effects.push_back(effect);
    c.xf = Mat2x3{};
    c.boundsHint = boundsHint;
    c.contentBounds = contentBounds;
    cl.effectCmds.push_back(c);
    cl.commands.push_back(Command{CmdKind::PushEffect, {}, static_cast<uint32_t>(cl.effectCmds.size() - 1)});
}

void addPopEffect(CommandList& cl) { cl.commands.push_back(Command{CmdKind::PopEffect, {}, 0}); }

EffectInstance dropShadow(float ox, float oy, float blur, Color color) {
    EffectInstance e{};
    e.kind = EffectKind::DropShadow;
    e.params[0] = ox;
    e.params[1] = oy;
    e.params[2] = blur;
    e.params[3] = 0;
    e.params[4] = 1;
    e.params[5] = 0;
    e.colors[0] = color;
    e.enabled = 1;
    return e;
}

/// Draws `cl` at 1x, checks the clip counts and the picture against SW; returns the GL pixels.
std::vector<uint8_t> checkAgainstSw(const CommandList& cl, const char* what, const char* slug,
                                    uint32_t wantAnalytic, AssetResolver* assets = nullptr,
                                    const IrView* ir = nullptr) {
    Session s(kW, kH);
    const std::vector<uint8_t> gl = s.render(cl, kClear, nullptr, assets, ir);
    const Session::ClipCounts counts = s.clipCounts(cl, nullptr, assets, ir);
    CHECK(counts.analytic == wantAnalytic);
    CHECK(counts.maskLayers == 0);
    CHECK(counts.fallbacks == 0);
    dump(gl, slug);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, kClear, nullptr, assets, ir);
    CHECK(matches(gl, sw, kW, kH, what));
    return gl;
}

/// A golden UI fixture's command list at time `t`, its whole picture inside one rounded clip.
struct Fixture {
    std::vector<uint8_t> ir;  // the document reads it for as long as it lives
    std::unique_ptr<FileAssetResolver> assets;
    std::unique_ptr<Document> doc;
    CommandList list;

    Fixture(const char* name, int w, int h, float t, float radius) {
        const std::string dir = XT_GOLDEN_UI_DIR;
        std::ifstream in(dir + "/" + name, std::ios::binary);
        REQUIRE(in.good());
        const std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        compile::CompileOptions options;
        options.emitMeta = false;
        options.rootDir = dir;
        std::vector<compile::SourceFile> files{{name, source}};
        compile::CompileResult result = compile::compileDocumentFromStrings(files, name, options);
        REQUIRE(result.ok());
        ir = std::move(result.ir);
        assets = std::make_unique<FileAssetResolver>(dir, sw::decodeImageBytes);
        REQUIRE(Document::load(IrBlob{ir.data(), ir.size()}, false, nullptr, assets.get(), doc) ==
                Status::kOk);
        doc->setViewport(Rect{0, 0, static_cast<float>(w), static_cast<float>(h)});
        doc->update(0);
        if (t > 0) {
            doc->update(t);
        }
        doc->paint(list);
        list.viewport = Rect{0, 0, static_cast<float>(w), static_cast<float>(h)};
        list.devicePixelRatio = 1;
        const float fw = static_cast<float>(w), fh = static_cast<float>(h);
        pushRoundedClip(list, Rect{fw * 0.06f, fh * 0.06f, fw * 0.88f, fh * 0.88f}, radius, radius, radius,
                        radius, Mat2x3{}, 0);
        addPopClip(list);
    }
};

}  // namespace

TEST_CASE("gl analytic clip: solid, gradient, coverage and box-shadow leaves in a rounded clip (T-719)") {
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{12.5f, 10.25f, 220, 136}, 28, 8, 40, 0);
    addFillRect(cl, 0, 0, kW, kH, Color{0.15f, 0.35f, 0.8f, 1});  // edge to edge: every corner is hit
    fillGradient(cl, Rect{0, 0, 130, 70}, Color{1, 0.2f, 0.1f, 1}, Color{1, 0.9f, 0.2f, 0.4f});
    // A concave L (drawn from the coverage atlas) running out of the clip's bottom-left corner.
    addFill(cl,
            addPolygon(
                cl, {Vec2{0, 90}, Vec2{90, 90}, Vec2{90, 120}, Vec2{40, 120}, Vec2{40, 160}, Vec2{0, 160}}),
            Color{0.2f, 0.9f, 0.5f, 0.9f});
    addBoxShadow(cl, Rect{150, 90, 70, 50}, 8, Vec2{6, 8}, 10, Color{0, 0, 0, 0.8f});
    addFillRect(cl, 150, 90, 70, 50, Color{0.9f, 0.9f, 0.95f, 1});
    addPopClip(cl);
    const std::vector<uint8_t> gl = checkAgainstSw(cl, "leaves in a rounded clip (1x)", "leaves", 1);
    // Outside the shape nothing was drawn: the clear colour at the four corners of its box (the
    // square bottom-left corner excepted) and beyond the box.
    const uint8_t* tl = pixel(gl, kW, 14, 12);
    const uint8_t* br = pixel(gl, kW, 230, 144);
    const uint8_t* out = pixel(gl, kW, 250, 80);
    for (const uint8_t* p : {tl, br, out}) {
        CHECK(p[0] <= 30);
        CHECK(p[2] <= 34);
    }
    const uint8_t* bl = pixel(gl, kW, 14, 144);  // radius 0: inside
    CHECK(bl[1] > 150);
}

TEST_CASE("gl analytic clip: a coverage-quad gradient and an outlined box shadow in a rounded clip (T-719)") {
    // The two leaf programs the case above does not reach: kCoverageGradientFrag (a concave gradient
    // fill) and kBoxShadowMultiFrag (a shadow of an outline ring plus the box, T-680).
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{16, 12, 224, 136}, 44);
    addFillRect(cl, 0, 0, kW, kH, Color{0.85f, 0.85f, 0.8f, 1});
    FillCmd f{};
    f.path =
        addPolygon(cl, {Vec2{0, 0}, Vec2{120, 0}, Vec2{120, 40}, Vec2{50, 40}, Vec2{50, 110}, Vec2{0, 110}});
    f.paint.kind = PaintKind::Linear;
    f.paint.color = Color{1, 1, 1, 1};
    f.paint.p0 = Vec2{0, 0};
    f.paint.p1 = Vec2{120, 110};
    f.paint.stopOffset = static_cast<uint32_t>(cl.stops.size());
    cl.stops.push_back(GradientStop{0.0f, Color{0.9f, 0.1f, 0.3f, 1}});
    cl.stops.push_back(GradientStop{1.0f, Color{0.1f, 0.3f, 0.9f, 0.7f}});
    f.paint.stopCount = 2;
    f.rule = FillRule::NonZero;
    f.antialias = 1;
    cl.fills.push_back(f);
    cl.commands.push_back(Command{CmdKind::FillPath, {}, static_cast<uint32_t>(cl.fills.size() - 1)});
    // The outline's outer edge, its inner edge (sign -1) and the border box (sign +1), bottom right.
    BoxShadowCmd& bs =
        addBoxShadow(cl, Rect{150, 80, 96, 70}, 14, Vec2{-6, -8}, 12, Color{0.1f, 0, 0.3f, 0.9f});
    bs.extraCount = 2;
    bs.extra[0].rect = Rect{156, 86, 84, 58};
    bs.extra[1].rect = Rect{162, 92, 72, 46};
    for (int i = 0; i < 4; ++i) {
        bs.extra[0].radii[i] = 8;
        bs.extra[1].radii[i] = 2;
    }
    bs.extra[0].sign = -1;
    bs.extra[1].sign = 1;
    addPopClip(cl);
    checkAgainstSw(cl, "coverage gradient and outlined box shadow in a rounded clip (1x)", "coverage_multi",
                   1);
}

TEST_CASE("gl analytic clip: images in a rounded clip (T-719)") {
    ImageResolver resolver(patterned(96, 64));
    IrView view{};
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{20, 14, 200, 130}, 30);
    addImage(cl, Rect{0, 0, 256, 160});  // magnified, edge to edge
    addImage(cl, Rect{-48, -32, 96, 64}, mul(translation(Vec2{190, 120}), rotationDeg(25.0f)));
    addPopClip(cl);
    checkAgainstSw(cl, "images in a rounded clip (1x)", "images", 1, &resolver, &view);
}

TEST_CASE("gl analytic clip: rotated, scaled and skewed clips (T-719)") {
    CommandList cl = makeList(kW, kH);
    const Mat2x3 rot = mul(translation(Vec2{62, 78}), rotationDeg(-17.0f));
    pushRoundedClip(cl, Rect{-45, -55, 90, 110}, 22, 6, 30, 14, rot);
    addFillRect(cl, 0, 0, kW, kH, Color{0.9f, 0.3f, 0.2f, 1});
    fillGradient(cl, Rect{-45, -55, 90, 60}, Color{1, 1, 1, 0.9f}, Color{0.2f, 0.2f, 1, 0.9f}, rot);
    addPopClip(cl);
    const Mat2x3 scaled = mul(translation(Vec2{150, 44}), scaling(Vec2{1.7f, 0.6f}));
    pushRoundedClip(cl, Rect{-30, -50, 60, 100}, 18, scaled);
    addFillRect(cl, 0, 0, kW, kH, Color{0.2f, 0.8f, 0.4f, 1});
    addPopClip(cl);
    const Mat2x3 skew = mul(translation(Vec2{176, 116}), Mat2x3{1, 0.25f, -0.45f, 1, 0, 0});
    pushRoundedClip(cl, Rect{-55, -30, 110, 60}, 20, skew);
    addFillRect(cl, 0, 0, kW, kH, Color{0.3f, 0.5f, 0.95f, 1});
    fillGradient(cl, Rect{-55, -30, 110, 60}, Color{1, 1, 0, 0.6f}, Color{1, 0, 1, 0.6f}, skew);
    addPopClip(cl);
    checkAgainstSw(cl, "rotated / scaled / skewed rounded clips (1x)", "transformed", 3);
}

TEST_CASE("gl analytic clip: two and three nested clips (T-719)") {
    CommandList two = makeList(kW, kH);
    pushRoundedClip(two, Rect{10, 8, 236, 144}, 36);
    addFillRect(two, 0, 0, kW, kH, Color{0.2f, 0.25f, 0.6f, 1});
    const Mat2x3 rot = mul(translation(Vec2{128, 80}), rotationDeg(31.0f));
    pushRoundedClip(two, Rect{-110, -50, 220, 100}, 26, rot);
    addFillRect(two, 0, 0, kW, kH, Color{0.9f, 0.6f, 0.2f, 1});
    CommandList three = two;  // the same two levels, and a third inside them
    addPopClip(two);
    addPopClip(two);
    checkAgainstSw(two, "two nested rounded clips (1x)", "nested2", 2);
    pushRoundedClip(three, Rect{60, 30, 150, 110}, 40, 4, 40, 4);
    fillGradient(three, Rect{0, 0, kW, kH}, Color{1, 1, 1, 1}, Color{0.1f, 0.6f, 0.3f, 1});
    addPopClip(three);
    addPopClip(three);
    addPopClip(three);
    checkAgainstSw(three, "nested rounded clips (1x)", "nested3", 3);
}

TEST_CASE("gl analytic clip: a group's result inside the clip is not clipped by the shape (T-719)") {
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{24, 18, 200, 120}, 30);
    addFillRect(cl, 0, 0, kW, kH, Color{0.25f, 0.25f, 0.3f, 1});
    // A card running out of the clip's bottom-right corner: the card (a leaf) is cut by the shape;
    // its glow is the effect group's result and is not (W1 item 3).
    addPushEffect(cl, dropShadow(10, 12, 8, Color{1, 0.85f, 0.2f, 1}), Rect{120, 60, 136, 100},
                  Rect{150, 80, 90, 60});
    addFillRect(cl, 150, 80, 90, 60, Color{0.9f, 0.4f, 0.1f, 1});
    addPopEffect(cl);
    addPushLayer(cl, 0.6f, BlendMode::Normal, Rect{0, 0, 120, 80});
    addFillRect(cl, 0, 0, 110, 70, Color{0.1f, 0.5f, 0.9f, 1});
    addFillRect(cl, 30, 20, 60, 60, Color{0.9f, 0.9f, 0.1f, 1});
    addPopLayer(cl);
    addPopClip(cl);
    const std::vector<uint8_t> gl = checkAgainstSw(cl, "group results in a rounded clip (1x)", "groups", 1);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, kClear);
    // The glow past the rounded corner: (220, 134) lies inside the box and outside the bottom-right
    // arc (centre 194,108, radius 30: distance 36.8). The mask layer of before left the clear colour.
    const uint8_t* g = pixel(gl, kW, 220, 134);
    const uint8_t* w = pixel(sw, kW, 220, 134);
    MESSAGE("glow past the corner: gl " << int(g[0]) << "," << int(g[1]) << "," << int(g[2]) << " sw "
                                        << int(w[0]) << "," << int(w[1]) << "," << int(w[2]));
    CHECK(g[0] > 150);
    CHECK(std::abs(int(g[0]) - int(w[0])) <= 16);
    CHECK(std::abs(int(g[1]) - int(w[1])) <= 16);
    CHECK(std::abs(int(g[2]) - int(w[2])) <= 16);
    // Outside the clip's box (T-720: the analytic level leaves the scissor alone): (229, 125).
    const uint8_t* o = pixel(gl, kW, 229, 125);
    const uint8_t* os = pixel(sw, kW, 229, 125);
    CHECK(os[0] > 150);
    CHECK(std::abs(int(o[0]) - int(os[0])) <= 16);
}

TEST_CASE("gl analytic clip: a drop shadow overflows a rotated rounded clip and its box as in SW (T-719)") {
    CommandList cl = makeList(kW, kH);
    const Mat2x3 rot = mul(translation(Vec2{100, 76}), rotationDeg(20.0f));
    pushRoundedClip(cl, Rect{-60, -40, 120, 80}, 18, rot);
    addFillRect(cl, 0, 0, kW, kH, Color{0.25f, 0.25f, 0.3f, 1});
    // A card under the clip's bottom-right corner with a wide, offset glow (hint: the whole glow).
    addPushEffect(cl, dropShadow(26, 18, 10, Color{1, 0.85f, 0.2f, 1}), Rect{90, 40, 166, 120},
                  Rect{120, 60, 90, 60});
    addFillRect(cl, 120, 60, 90, 60, Color{0.2f, 0.3f, 0.9f, 1});
    addPopEffect(cl);
    addPopClip(cl);
    const std::vector<uint8_t> gl =
        checkAgainstSw(cl, "drop shadow overflowing a rotated rounded clip (1x)", "overflow", 1);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, kClear);
    // Outside the clip's box (the rotated rectangle's corners): where SW shows the glow brightest,
    // GL shows it too, and hardly any pixel there differs from SW.
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    for (const Vec2 c : {Vec2{-60, -40}, Vec2{60, -40}, Vec2{60, 40}, Vec2{-60, 40}}) {
        const Vec2 p = apply(rot, c);
        bx0 = std::min(bx0, p.x);
        by0 = std::min(by0, p.y);
        bx1 = std::max(bx1, p.x);
        by1 = std::max(by1, p.y);
    }
    int bestX = -1, bestY = -1, bestR = -1;
    uint32_t outsideDiff = 0, outsidePx = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const float cx = static_cast<float>(x) + 0.5f, cy = static_cast<float>(y) + 0.5f;
            if (cx >= bx0 - 1.0f && cx <= bx1 + 1.0f && cy >= by0 - 1.0f && cy <= by1 + 1.0f) {
                continue;
            }
            ++outsidePx;
            const uint8_t* g = pixel(gl, kW, x, y);
            const uint8_t* w = pixel(sw, kW, x, y);
            const int d = std::max({std::abs(int(g[0]) - int(w[0])), std::abs(int(g[1]) - int(w[1])),
                                    std::abs(int(g[2]) - int(w[2]))});
            outsideDiff += d > 16 ? 1u : 0u;
            if (int(w[0]) > bestR) {
                bestR = int(w[0]);
                bestX = x;
                bestY = y;
            }
        }
    }
    REQUIRE(bestX >= 0);
    const uint8_t* g = pixel(gl, kW, bestX, bestY);
    MESSAGE("glow outside the clip's box at " << bestX << "," << bestY << ": sw " << bestR << ", gl "
                                              << int(g[0]) << "; " << outsideDiff << " of " << outsidePx
                                              << " px outside the box differ");
    CHECK(bestR > 150);  // the glow does reach past the box
    CHECK(std::abs(int(g[0]) - bestR) <= 16);
    CHECK(outsideDiff <= outsidePx / 100u);
}

TEST_CASE("gl analytic clip: the edge against the supersampled circular-cornered rectangle (T-719)") {
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{20.3f, 16.6f, 200.4f, 120.7f}, 34);
    addFillRect(cl, 0, 0, kW, kH, Color{1, 1, 1, 1});
    addPopClip(cl);
    Session s(kW, kH);
    const std::vector<uint8_t> gl = s.render(cl, Color{0, 0, 0, 1});
    CHECK(s.clipCounts(cl).analytic == 1);
    CHECK(s.renderer.frameStats().passes == 1);  // no mask layer: the surface pass alone
    dump(gl, "edge");
    // The true shape (circular arcs), 16 x 16 samples per pixel (as the wgpu case: at most 24 / 255).
    const float bx = 20.3f, by = 16.6f, bw = 200.4f, bh = 120.7f, br = 34.0f;
    auto inside = [&](float x, float y) {
        const float qx = std::fabs(x - (bx + bw * 0.5f)) - (bw * 0.5f - br);
        const float qy = std::fabs(y - (by + bh * 0.5f)) - (bh * 0.5f - br);
        if (qx > 0.0f && qy > 0.0f) {
            return qx * qx + qy * qy <= br * br;
        }
        return qx <= br && qy <= br;
    };
    int worst = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            int hits = 0;
            for (int sy = 0; sy < 16; ++sy) {
                for (int sx = 0; sx < 16; ++sx) {
                    hits += inside(static_cast<float>(x) + (static_cast<float>(sx) + 0.5f) / 16.0f,
                                   static_cast<float>(y) + (static_cast<float>(sy) + 0.5f) / 16.0f)
                                ? 1
                                : 0;
                }
            }
            const int truth = (hits * 255 + 128) / 256;
            worst = std::max(worst, std::abs(int(pixel(gl, kW, x, y)[0]) - truth));
        }
    }
    MESSAGE("against the supersampled circular-cornered rectangle: analytic max " << worst);
    CHECK(worst <= 24);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, Color{0, 0, 0, 1});
    CHECK(matches(gl, sw, kW, kH, "one rounded clip, analytic (1x)"));
}

TEST_CASE("gl analytic clip: minified, BC7 and particle draws in a rounded clip (T-719)") {
    // The leaf programs no hand-written list reaches -- kTextureLodFrag (ui_image_minify),
    // kTextureStraightFrag (ui_image_bc7: compressed blocks keep straight alpha) and the two particle
    // programs (ui_particles at 0.5 s) -- each fixture's whole picture inside one rounded clip.
    struct Case {
        const char* file;
        const char* slug;
        int w, h;
        float t;
        Color clear;
    };
    const Case cases[] = {
        {"ui_image_minify.xt", "minify", 256, 256, 0.0f, Color{1, 1, 1, 1}},
        {"ui_image_bc7.xt", "bc7", 320, 208, 0.0f, Color{0xc8 / 255.0f, 0xd0 / 255.0f, 0xd8 / 255.0f, 1}},
        {"ui_particles.xt", "particles", 256, 256, 0.5f, Color{1, 1, 1, 1}},
    };
    for (const Case& c : cases) {
        CAPTURE(c.file);
        Fixture fx(c.file, c.w, c.h, c.t, 48.0f);
        Session s(c.w, c.h);
        std::vector<uint8_t> gl;
        for (int frame = 0; frame < 2; ++frame) {  // the second frame samples the mip chains
            gl = s.render(fx.list, c.clear, nullptr, fx.assets.get(), &fx.doc->ir());
        }
        const Session::ClipCounts counts = s.clipCounts(fx.list, nullptr, fx.assets.get(), &fx.doc->ir());
        CHECK(counts.analytic >= 1);
        CHECK(counts.maskLayers == 0);
        dump(gl, c.slug);
        const std::vector<uint8_t> sw =
            renderSw(fx.list, c.w, c.h, c.clear, nullptr, fx.assets.get(), &fx.doc->ir());
        CHECK(matches(gl, sw, c.w, c.h, c.file));
        // The clip's top-left corner pixel (inside its box, outside the arc) keeps the clear colour.
        const int cx = static_cast<int>(static_cast<float>(c.w) * 0.06f) + 3;
        const int cy = static_cast<int>(static_cast<float>(c.h) * 0.06f) + 3;
        const uint8_t* p = pixel(gl, c.w, cx, cy);
        CHECK(std::abs(int(p[0]) - int(std::lround(c.clear.r * 255.0f))) <= 1);
        CHECK(std::abs(int(p[1]) - int(std::lround(c.clear.g * 255.0f))) <= 1);
        CHECK(std::abs(int(p[2]) - int(std::lround(c.clear.b * 255.0f))) <= 1);
    }
}

#ifndef XT_GL_NO_TEXT

TEST_CASE("gl analytic clip: text in a rounded clip (T-719)") {
    text::FtTextShaper shaper;
    REQUIRE(shaper.addFontFile(std::string(XT_TEST_FONTS_DIR) + "/NotoSans-Regular.subset.ttf"));
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{30, 40, 150, 60}, 30);
    addFillRect(cl, 0, 0, kW, kH, Color{0.9f, 0.9f, 0.85f, 1});
    ShapeParams params;
    static const std::vector<std::string_view> family = {"Noto Sans"};
    params.font = FontRequest{std::span<const std::string_view>(family), 400, false};
    for (int line = 0; line < 3; ++line) {
        const float size = line == 1 ? 40.0f : 16.0f;  // the atlas / hinted path and a large run
        params.size = size;
        ShapeResult shaped;
        shaper.shape("Clipped text run 1x", params, shaped);
        REQUIRE(!shaped.runs.empty());
        GlyphsCmd run{};
        run.font = shaped.runs[0].font;
        run.size = size;
        run.color = Color{0.1f, 0.1f, 0.3f, 1};
        run.glyphOffset = static_cast<uint32_t>(cl.glyphs.size());
        float pen = 4.0f;
        const float baseline = 52.0f + 22.0f * static_cast<float>(line);
        for (const ShapedGlyph& g : shaped.glyphs) {
            cl.glyphs.push_back(g.glyphId);
            cl.glyphPos.push_back(Vec2{pen + g.xOffset, baseline + g.yOffset});
            pen += g.xAdvance;
        }
        run.glyphCount = static_cast<uint32_t>(cl.glyphs.size()) - run.glyphOffset;
        cl.glyphRuns.push_back(run);
        cl.commands.push_back(
            Command{CmdKind::DrawGlyphs, {}, static_cast<uint32_t>(cl.glyphRuns.size() - 1)});
    }
    addPopClip(cl);

    Session s(kW, kH);
    text::GlyphAtlas atlas(shaper, 512);
    s.renderer.setGlyphAtlas(&atlas);
    const std::vector<uint8_t> gl = s.render(cl, kClear, &shaper);
    const Session::ClipCounts counts = s.clipCounts(cl, &shaper);
    CHECK(counts.analytic == 1);
    CHECK(counts.maskLayers == 0);
    s.renderer.setGlyphAtlas(nullptr);
    dump(gl, "text");
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, kClear, &shaper);
    CHECK(matches(gl, sw, kW, kH, "text in a rounded clip (1x)"));
    // Glyphs left of the clip's rounded start are not drawn.
    const uint8_t* p = pixel(gl, kW, 8, 60);
    CHECK(p[0] <= 30);
}

#endif  // XT_GL_NO_TEXT
