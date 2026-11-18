// T-718 (Docs/design/10-backends.md §10.16 W1, Docs/tasks/T-718.md "測試"): rounded clips drawn as
// analytic clip levels by xt_wgpu at 1x, against xt::sw (15 §15.3's tolerance: channel 16, 3 % of
// the pixels) -- solid, gradient, image, coverage and box-shadow leaves, text, transformed and
// nested clips, and groups (layer, effect) whose results are not clipped.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "doctest.h"
#include "rounded_rect.h"  // Code/Runtime/core/src: appendRoundedRectPath
#include "test_wgpu_draw_util.h"
#include "xt/command_list.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/wgpu/wgpu_device.h"
#ifdef XT_GOLDEN_TEXT
#include "xt/text/ft_text_shaper.h"
#include "xt/text/glyph_atlas.h"
#include "xt/text_shaper.h"
#endif

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kW = 256, kH = 160;

void pushRoundedClip(CommandList& cl, Rect box, float tl, float tr, float br, float bl,
                     Mat2x3 xf = Mat2x3{}) {
    ClipPathCmd c{};
    c.xf = xf;
    c.path = emptyPath(cl);
    appendRoundedRectPath(box, tl, tr, br, bl, cl.verbs, cl.points);
    c.path.verbCount = static_cast<uint32_t>(cl.verbs.size()) - c.path.verbOffset;
    c.path.pointCount = static_cast<uint32_t>(cl.points.size()) - c.path.pointOffset;
    c.rule = FillRule::NonZero;
    cl.clipPaths.push_back(c);
    cl.commands.push_back(Command{CmdKind::PushClipPath, {}, static_cast<uint32_t>(cl.clipPaths.size() - 1)});
}

void pushRoundedClip(CommandList& cl, Rect box, float r, Mat2x3 xf = Mat2x3{}) {
    pushRoundedClip(cl, box, r, r, r, r, xf);
}

PathRef rectPath(CommandList& cl, Rect r) {
    return addPolygon(
        cl, {Vec2{r.x, r.y}, Vec2{r.x + r.w, r.y}, Vec2{r.x + r.w, r.y + r.h}, Vec2{r.x, r.y + r.h}});
}

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

void addBoxShadow(CommandList& cl, Rect r, float radius, Vec2 offset, float blur, Color color) {
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
}

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

/// xt_wgpu as a device that does not report analytic clips (a host's device that has not caught up).
class PlainDevice : public wgpu::WgpuDevice {
public:
    bool supportsAnalyticClip() const override { return false; }
};

/// `cl` built for and drawn by a PlainDevice at 1x: the mask layers of before.
std::vector<uint8_t> drawWithoutAnalyticClip(const CommandList& cl, Color clear, uint32_t& maskLayers,
                                             uint32_t& passes) {
    Session s(kW, kH);  // its target and readback
    PlainDevice device;
    std::string err;
    REQUIRE_MESSAGE(
        device.init(sharedInit().device, sharedInit().queue, WGPUTextureFormat_RGBA8Unorm, 1, err), err);
    device.resize(kW, kH);
    gpu::FrameBuilder builder;
    builder.init(device);
    builder.resize(kW, kH);
    builder.setEdgeFringe(true);
    gpu::GpuFrame frame;
    builder.build(cl, nullptr, nullptr, nullptr, frame);
    maskLayers = builder.lastClipStats().maskLayers;
    passes = static_cast<uint32_t>(frame.passes.size());
    CHECK(builder.lastBuildStats().analyticClips == 0);
    for (const gpu::DrawCmd& c : frame.cmds) {
        CHECK(c.clipParamIndex == 0);
    }
    device.setSurfaceClear(true, WGPUColor{clear.r * clear.a, clear.g * clear.a, clear.b * clear.a, clear.a});
    device.setSurface(s.target.view);
    device.submit(frame);
    device.setSurface(nullptr);
    CHECK_MESSAGE(std::string(device.lastError()).empty(), device.lastError());
    std::vector<uint8_t> px = s.read();
    builder.shutdown();
    device.shutdown();
    return px;
}

/// T-719: XT_ANALYTIC_CLIP_DUMP=<dir> writes a case's pixels to <dir>/wgpu_<slug>.rgba (raw RGBA8, rows
/// top-down), as Tests/gl/test_gl_analytic_clip.cpp does with gl_<slug>.rgba, for the GL / wgpu comparison.
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
    std::ofstream out(dir + "/wgpu_" + slug + ".rgba", std::ios::binary);
    out.write(reinterpret_cast<const char*>(px.data()), static_cast<std::streamsize>(px.size()));
}

/// Draws `cl` at 1x and checks it against SW; returns the wgpu pixels.
std::vector<uint8_t> checkAgainstSw(const CommandList& cl, const char* what, uint32_t wantAnalytic,
                                    AssetResolver* assets = nullptr, const IrView* ir = nullptr,
                                    float ratio = 0.03f) {
    Session s(kW, kH);
    const std::vector<uint8_t> gpu = s.render(cl, Color{0.1f, 0.1f, 0.12f, 1}, assets, ir);
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(s.renderer.stats().analyticClips == wantAnalytic);
    CHECK(s.renderer.stats().clipMaskLayers == 0);
    CHECK(s.renderer.stats().analyticClipFallbacks == 0);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, Color{0.1f, 0.1f, 0.12f, 1}, assets, ir);
    CHECK(matches(gpu, sw, kW, kH, what, ratio));
    return gpu;
}

}  // namespace

TEST_CASE("wgpu analytic clip: solid, gradient, coverage and box-shadow leaves in a rounded clip (T-718)") {
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
    const std::vector<uint8_t> gpu = checkAgainstSw(cl, "leaves in a rounded clip (1x)", 1);
    dump(gpu, "leaves");
    // Outside the shape nothing was drawn: the clear colour at the four corners of its box (the
    // square bottom-left corner excepted) and beyond the box.
    const uint8_t* tl = pixel(gpu, kW, 14, 12);
    const uint8_t* br = pixel(gpu, kW, 230, 144);
    const uint8_t* out = pixel(gpu, kW, 250, 80);
    for (const uint8_t* p : {tl, br, out}) {
        CHECK(p[0] <= 30);
        CHECK(p[2] <= 34);
    }
    const uint8_t* bl = pixel(gpu, kW, 14, 144);  // radius 0: inside
    CHECK(bl[1] > 150);
}

TEST_CASE("wgpu analytic clip: images in a rounded clip (T-718)") {
    ImageResolver resolver(patterned(96, 64));
    IrView view{};
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{20, 14, 200, 130}, 30);
    addImage(cl, Rect{0, 0, 256, 160});  // magnified, edge to edge
    addImage(cl, Rect{-48, -32, 96, 64}, mul(translation(Vec2{190, 120}), rotationDeg(25.0f)));
    addPopClip(cl);
    dump(checkAgainstSw(cl, "images in a rounded clip (1x)", 1, &resolver, &view), "images");
}

TEST_CASE("wgpu analytic clip: rotated, scaled and skewed clips (T-718)") {
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
    dump(checkAgainstSw(cl, "rotated / scaled / skewed rounded clips (1x)", 3), "transformed");
}

TEST_CASE("wgpu analytic clip: two and three nested clips (T-718)") {
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{10, 8, 236, 144}, 36);
    addFillRect(cl, 0, 0, kW, kH, Color{0.2f, 0.25f, 0.6f, 1});
    const Mat2x3 rot = mul(translation(Vec2{128, 80}), rotationDeg(31.0f));
    pushRoundedClip(cl, Rect{-110, -50, 220, 100}, 26, rot);
    addFillRect(cl, 0, 0, kW, kH, Color{0.9f, 0.6f, 0.2f, 1});
    pushRoundedClip(cl, Rect{60, 30, 150, 110}, 40, 4, 40, 4);
    fillGradient(cl, Rect{0, 0, kW, kH}, Color{1, 1, 1, 1}, Color{0.1f, 0.6f, 0.3f, 1});
    addPopClip(cl);
    addPopClip(cl);
    addPopClip(cl);
    dump(checkAgainstSw(cl, "nested rounded clips (1x)", 3), "nested3");
}

TEST_CASE("wgpu analytic clip: a group's result inside the clip is not clipped by the shape (T-718)") {
    const Color clear{0.1f, 0.1f, 0.12f, 1};
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{24, 18, 200, 120}, 30);
    addFillRect(cl, 0, 0, kW, kH, Color{0.25f, 0.25f, 0.3f, 1});
    // A card running out of the clip's bottom-right corner: the card (a leaf) is cut by the shape;
    // its glow is the effect group's result and is not (W1 item 3).
    addPushEffect(cl, {dropShadow(10, 12, 8, Color{1, 0.85f, 0.2f, 1})}, Rect{120, 60, 136, 100},
                  Rect{150, 80, 90, 60});
    addFillRect(cl, 150, 80, 90, 60, Color{0.9f, 0.4f, 0.1f, 1});
    addPopEffect(cl);
    addPushLayer(cl, 0.6f, BlendMode::Normal, Rect{0, 0, 120, 80});
    addFillRect(cl, 0, 0, 110, 70, Color{0.1f, 0.5f, 0.9f, 1});
    addFillRect(cl, 30, 20, 60, 60, Color{0.9f, 0.9f, 0.1f, 1});
    addPopLayer(cl);
    addPopClip(cl);
    Session s1(kW, kH);
    const std::vector<uint8_t> gpu = s1.render(cl, clear);
    dump(gpu, "groups");
    CHECK_MESSAGE(s1.error.empty(), s1.error);
    CHECK(s1.renderer.stats().analyticClips == 1);
    CHECK(s1.renderer.stats().clipMaskLayers == 0);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, clear);
    // Inside the clip's box (the scissor) the picture is SW's.
    uint32_t insideDiff = 0, outsideDiff = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const uint8_t* g = pixel(gpu, kW, x, y);
            const uint8_t* w = pixel(sw, kW, x, y);
            const int d = std::max({std::abs(int(g[0]) - int(w[0])), std::abs(int(g[1]) - int(w[1])),
                                    std::abs(int(g[2]) - int(w[2]))});
            const bool inBox = x >= 24 && x < 224 && y >= 18 && y < 138;
            (inBox ? insideDiff : outsideDiff) += d > 16 ? 1u : 0u;
        }
    }
    MESSAGE("group results vs sw: " << insideDiff << " px differ inside the clip's box, " << outsideDiff
                                    << " outside it");
    CHECK(insideDiff <= 200u * 120u * 3u / 100u);
    CHECK(outsideDiff <= (256u * 160u - 200u * 120u) * 3u / 100u);  // T-720: no scissor at the box
    // The glow past the rounded corner: (220, 134) lies inside the box and outside the bottom-right
    // arc (centre 194,108, radius 30: distance 36.8). The mask layer of before left the clear colour.
    const uint8_t* g = pixel(gpu, kW, 220, 134);
    const uint8_t* w = pixel(sw, kW, 220, 134);
    MESSAGE("glow past the corner: wgpu " << int(g[0]) << "," << int(g[1]) << "," << int(g[2]) << " sw "
                                          << int(w[0]) << "," << int(w[1]) << "," << int(w[2]));
    CHECK(g[0] > 150);
    CHECK(std::abs(int(g[0]) - int(w[0])) <= 16);
    CHECK(std::abs(int(g[1]) - int(w[1])) <= 16);
    CHECK(std::abs(int(g[2]) - int(w[2])) <= 16);
    // Outside the clip's box: T-720 (W1 item 1 as revised) leaves the scissor alone, so 1x draws the
    // glow there as xt::sw does; the 4x stencil level's scissor still cuts it (a separate case,
    // PENDING): (229, 125).
    Session s4(kW, kH, 4);
    const std::vector<uint8_t> gpu4 = s4.render(cl, clear);
    CHECK(s4.renderer.stats().analyticClips == 0);
    const uint8_t* o1 = pixel(gpu, kW, 229, 125);
    const uint8_t* o4 = pixel(gpu4, kW, 229, 125);
    const uint8_t* os = pixel(sw, kW, 229, 125);
    MESSAGE("outside the box: wgpu 1x " << int(o1[0]) << ", wgpu 4x " << int(o4[0]) << ", sw " << int(os[0]));
    CHECK(os[0] > 150);
    CHECK(std::abs(int(o1[0]) - int(os[0])) <= 16);
    CHECK(o4[0] <= 30);
    // And 1x against SW over the whole picture.
    CHECK(matches(gpu, sw, kW, kH, "group results in a rounded clip (1x)"));
}

TEST_CASE("wgpu analytic clip: a drop shadow overflows a rotated rounded clip and its box as in SW (T-720)") {
    // Docs/design/10-backends.md §10.16 W1, item 1 as revised by T-720: the analytic level leaves the
    // scissor alone, so an effect's shadow running out of a rotated rounded clip is not cut into an
    // axis-aligned rectangle at the clip's box (the War Table hand cards).
    const Color clear{0.1f, 0.1f, 0.12f, 1};
    CommandList cl = makeList(kW, kH);
    const Mat2x3 rot = mul(translation(Vec2{100, 76}), rotationDeg(20.0f));
    pushRoundedClip(cl, Rect{-60, -40, 120, 80}, 18, rot);
    addFillRect(cl, 0, 0, kW, kH, Color{0.25f, 0.25f, 0.3f, 1});
    // A card under the clip's bottom-right corner with a wide, offset glow (hint: the whole glow).
    addPushEffect(cl, {dropShadow(26, 18, 10, Color{1, 0.85f, 0.2f, 1})}, Rect{90, 40, 166, 120},
                  Rect{120, 60, 90, 60});
    addFillRect(cl, 120, 60, 90, 60, Color{0.2f, 0.3f, 0.9f, 1});
    addPopEffect(cl);
    addPopClip(cl);
    Session s(kW, kH);
    const std::vector<uint8_t> gpu = s.render(cl, clear);
    dump(gpu, "overflow");
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(s.renderer.stats().analyticClips == 1);
    CHECK(s.renderer.stats().clipMaskLayers == 0);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, clear);
    CHECK(matches(gpu, sw, kW, kH, "drop shadow overflowing a rotated rounded clip (1x)"));
    // The clip's box (device px): the rotated rectangle's corners.
    float bx0 = 1e9f, by0 = 1e9f, bx1 = -1e9f, by1 = -1e9f;
    for (const Vec2 c : {Vec2{-60, -40}, Vec2{60, -40}, Vec2{60, 40}, Vec2{-60, 40}}) {
        const Vec2 p = apply(rot, c);
        bx0 = std::min(bx0, p.x);
        by0 = std::min(by0, p.y);
        bx1 = std::max(bx1, p.x);
        by1 = std::max(by1, p.y);
    }
    // Outside the box: where SW shows the glow brightest, wgpu shows it too (the scissor of T-718
    // left the clear colour there), and outside the box hardly any pixel differs from SW.
    int bestX = -1, bestY = -1, bestR = -1;
    uint32_t outsideDiff = 0, outsidePx = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const float cx = static_cast<float>(x) + 0.5f, cy = static_cast<float>(y) + 0.5f;
            if (cx >= bx0 - 1.0f && cx <= bx1 + 1.0f && cy >= by0 - 1.0f && cy <= by1 + 1.0f) {
                continue;
            }
            ++outsidePx;
            const uint8_t* g = pixel(gpu, kW, x, y);
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
    const uint8_t* g = pixel(gpu, kW, bestX, bestY);
    MESSAGE("glow outside the clip's box at " << bestX << "," << bestY << ": sw " << bestR << ", wgpu "
                                              << int(g[0]) << "; " << outsideDiff << " of " << outsidePx
                                              << " px outside the box differ");
    CHECK(bestR > 150);  // the glow does reach past the box
    CHECK(std::abs(int(g[0]) - bestR) <= 16);
    CHECK(outsideDiff <= outsidePx / 100u);
}

TEST_CASE("wgpu analytic clip: a straight edge is the mask layer's edge, a corner is rounder (T-718)") {
    // The same clip through the analytic level and, on a device that does not report the
    // capability, through the mask layer of before.
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{20.3f, 16.6f, 200.4f, 120.7f}, 34);
    addFillRect(cl, 0, 0, kW, kH, Color{1, 1, 1, 1});
    addPopClip(cl);
    Session a(kW, kH);
    const std::vector<uint8_t> pa = a.render(cl, Color{0, 0, 0, 1});
    dump(pa, "edge");
    CHECK(a.renderer.stats().analyticClips == 1);
    uint32_t maskLayers = 0, passes = 0;
    const std::vector<uint8_t> pm = drawWithoutAnalyticClip(cl, Color{0, 0, 0, 1}, maskLayers, passes);
    CHECK(maskLayers == 1);
    MESSAGE("passes: analytic " << a.renderer.stats().passes << ", mask layer " << passes);
    CHECK(a.renderer.stats().passes + 3 <= passes);  // the shape, the content and the MaskApply targets
    // Rows / columns through the straight parts of the four sides, and the four corners.
    int straightMax = 0, cornerMax = 0;
    for (int y = 0; y < kH; ++y) {
        for (int x = 0; x < kW; ++x) {
            const bool straight = (x >= 60 && x <= 180) || (y >= 56 && y <= 96);
            const int d = std::abs(int(pixel(pa, kW, x, y)[0]) - int(pixel(pm, kW, x, y)[0]));
            int& worst = straight ? straightMax : cornerMax;
            worst = std::max(worst, d);
        }
    }
    MESSAGE("analytic vs mask layer: straight edges max " << straightMax << ", corners max " << cornerMax);
    CHECK(straightMax <= 1);  // W1 item 4: <= 1 / 255 on the straight sides
    CHECK(cornerMax <= 64);   // at most about 0.2 px at the arcs
    // The true shape (circular arcs), 16 x 16 samples per pixel: the analytic edge is the nearer one.
    const float bx = 20.3f, by = 16.6f, bw = 200.4f, bh = 120.7f, br = 34.0f;
    auto inside = [&](float x, float y) {
        const float qx = std::fabs(x - (bx + bw * 0.5f)) - (bw * 0.5f - br);
        const float qy = std::fabs(y - (by + bh * 0.5f)) - (bh * 0.5f - br);
        if (qx > 0.0f && qy > 0.0f) {
            return qx * qx + qy * qy <= br * br;
        }
        return qx <= br && qy <= br;
    };
    int analyticWorst = 0, maskWorst = 0;
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
            analyticWorst = std::max(analyticWorst, std::abs(int(pixel(pa, kW, x, y)[0]) - truth));
            maskWorst = std::max(maskWorst, std::abs(int(pixel(pm, kW, x, y)[0]) - truth));
        }
    }
    MESSAGE("against the supersampled circular-cornered rectangle: analytic max "
            << analyticWorst << ", mask layer max " << maskWorst);
    CHECK(analyticWorst <= 24);
    CHECK(analyticWorst < maskWorst);
    // Both within the tolerance of SW.
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, Color{0, 0, 0, 1});
    CHECK(matches(pa, sw, kW, kH, "one rounded clip, analytic (1x)"));
    CHECK(matches(pm, sw, kW, kH, "one rounded clip, mask layer (1x)"));
}

TEST_CASE("wgpu analytic clip: 4x keeps its stencil clip and matches SW (T-718)") {
    CommandList cl = makeList(kW, kH);
    pushRoundedClip(cl, Rect{12.5f, 10.25f, 220, 136}, 28, 8, 40, 0);
    addFillRect(cl, 0, 0, kW, kH, Color{0.15f, 0.35f, 0.8f, 1});
    fillGradient(cl, Rect{0, 0, 130, 70}, Color{1, 0.2f, 0.1f, 1}, Color{1, 0.9f, 0.2f, 0.4f});
    addPopClip(cl);
    Session s(kW, kH, 4);
    const std::vector<uint8_t> gpu = s.render(cl, Color{0.1f, 0.1f, 0.12f, 1});
    CHECK(s.error.empty());
    CHECK(s.renderer.stats().analyticClips == 0);
    CHECK(s.renderer.stats().analyticClipFallbacks == 0);
    const std::vector<uint8_t> sw = renderSw(cl, kW, kH, Color{0.1f, 0.1f, 0.12f, 1});
    CHECK(matches(gpu, sw, kW, kH, "rounded clip (4x)"));
}

#ifdef XT_GOLDEN_TEXT

TEST_CASE("wgpu analytic clip: text in a rounded clip (T-718)") {
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
    s.renderer.setClearColor(0.1f, 0.1f, 0.12f, 1);
    s.renderer.render(cl, &shaper, nullptr, s.target.view);
    CHECK_MESSAGE(std::string(s.renderer.lastError()).empty(), s.renderer.lastError());
    CHECK(s.renderer.stats().analyticClips == 1);
    CHECK(s.renderer.stats().clipMaskLayers == 0);
    const std::vector<uint8_t> gpu = s.read();
    dump(gpu, "text");
    s.renderer.setGlyphAtlas(nullptr);

    std::vector<uint8_t> sw(static_cast<size_t>(kW) * kH * 4, 0);
    sw::SwRenderer swr;
    swr.setTarget(sw::SwTarget{sw.data(), kW, kH, kW * 4});
    swr.clear(Color{0.1f, 0.1f, 0.12f, 1});
    swr.render(cl, &shaper, nullptr, nullptr);
    CHECK(matches(gpu, sw, kW, kH, "text in a rounded clip (1x)"));
    // Glyphs left of the clip's rounded start are not drawn.
    const uint8_t* p = pixel(gpu, kW, 8, 60);
    CHECK(p[0] <= 30);
}

#endif  // XT_GOLDEN_TEXT
