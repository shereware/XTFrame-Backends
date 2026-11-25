// T-438 part b (Docs/tasks/T-438.md item 14, Docs/design/10-backends.md §10.10.3): effect programs,
// composite modes, particles, the Direct target mode and custom effects through xt_wgpu, compared
// with xt::sw at 15 §15.3's GPU tolerance (Outline 5 %, as Tests/gl).

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "doctest.h"
#include "test_wgpu_draw_util.h"
#include "wgpu_internal.h"  // detail::wrapCustomModule (T-892; Code/Runtime/wgpu/src)
#include "xt/compile/compile.h"
#include "xt/compile/xtfx.h"
#include "xt/document.h"
#include "xt/perf.h"
#include "xt/sw/sw_png.h"
#include "xt/xtfx_reserved.h"

using namespace xt;
using namespace xt::wgpu::test;

namespace {

constexpr int kS = 96;

/// One effect over a red square in the middle of a `kS` x `kS` surface.
CommandList effectList(std::initializer_list<EffectInstance> effects, float margin = 24) {
    CommandList cl = makeList(kS, kS);
    const Rect content{30, 30, 36, 36};
    const Rect bounds{content.x - margin, content.y - margin, content.w + 2 * margin, content.h + 2 * margin};
    addPushEffect(cl, effects, bounds, content);
    addFillRect(cl, content.x, content.y, content.w, content.h, Color{1, 0.1f, 0.1f, 1});
    addFillRect(cl, 40, 40, 16, 16, Color{0.1f, 0.3f, 1, 0.6f});
    addPopEffect(cl);
    return cl;
}

void addShader(CommandList& cl, AssetId asset, const std::string& glsl, const std::vector<std::string>& names,
               const std::vector<uint8_t>& types, EffectInstance fallback) {
    ShaderRef ref{};
    ref.asset = asset;
    ref.sourceOffset = static_cast<uint32_t>(cl.shaderText.size());
    ref.glslOffset = static_cast<uint32_t>(cl.shaderText.size());
    ref.glslLength = static_cast<uint32_t>(glsl.size());
    cl.shaderText.insert(cl.shaderText.end(), glsl.begin(), glsl.end());
    ref.namesOffset = static_cast<uint32_t>(cl.shaderText.size());
    for (const std::string& n : names) {
        cl.shaderText.insert(cl.shaderText.end(), n.begin(), n.end());
        cl.shaderText.push_back('\0');
    }
    ref.namesLength = static_cast<uint32_t>(cl.shaderText.size()) - ref.namesOffset;
    ref.paramOffset = static_cast<uint32_t>(cl.shaderParams.size());
    ref.paramCount = static_cast<uint32_t>(types.size());
    for (uint8_t t : types) {
        ShaderParamDesc d{};
        d.type = t;
        cl.shaderParams.push_back(d);
    }
    ref.fallback = fallback;
    cl.shaders.push_back(ref);
}

EffectInstance customFx(AssetId asset, std::initializer_list<float> params,
                        Color color0 = Color{1, 1, 1, 1}) {
    EffectInstance e = effect(EffectKind::Custom, params, color0);
    e.shader = asset;
    return e;
}

std::string glslOf(const char* xtfx) {
    compile::XtfxResult r = compile::compileXtfx(xtfx);
    REQUIRE_MESSAGE(r.ok, r.error.message);
    return r.glsl;
}

int near(int value, int expected) { return std::abs(value - expected); }

}  // namespace

TEST_CASE("wgpu effects: blur, drop shadow, glow, color adjust, shine and outline match SW") {
    struct Case {
        const char* name;
        CommandList list;
        float ratio;
    };
    const Case cases[] = {
        {"blur", effectList({effect(EffectKind::Blur, {5, 5})}), 0.03f},
        {"drop shadow", effectList({effect(EffectKind::DropShadow, {6, 6, 4, 2, 0.8f})}), 0.03f},
        {"inset shadow", effectList({effect(EffectKind::DropShadow, {4, 4, 6, 2, 1, 1})}, 0), 0.03f},
        {"glow outer", effectList({effect(EffectKind::Glow, {8, 1, 0}, Color{1, 1, 0, 1})}), 0.03f},
        {"glow inner", effectList({effect(EffectKind::Glow, {6, 1, 1}, Color{0, 1, 0, 1})}), 0.03f},
        {"color adjust", effectList({effect(EffectKind::ColorAdjust, {0.1f, 1.2f, 0.4f, 40, 0.9f})}), 0.03f},
        {"shine", effectList({effect(EffectKind::Shine, {0.5f, 0.3f, 0.2f, 30, 0.8f}, Color{1, 1, 1, 1})}),
         0.03f},
        {"outline", effectList({effect(EffectKind::Outline, {3, 1, 0}, Color{1, 1, 1, 1})}), 0.05f},
        {"stack",
         effectList({effect(EffectKind::Blur, {2, 2}), effect(EffectKind::DropShadow, {3, 3, 2, 0, 1})}),
         0.03f},
    };
    Session s(kS, kS);
    for (const Case& c : cases) {
        CAPTURE(c.name);
        PerfTracker perf;
        perf.setEnabled(true);
        perf.beginFrame(0.0f);
        const std::vector<uint8_t> gpu = s.render(c.list, Color{0, 0, 0, 1}, nullptr, nullptr, &perf);
        CHECK_MESSAGE(s.error.empty(), s.error);
        CHECK(matches(gpu, renderSw(c.list, kS, kS), kS, kS, c.name, c.ratio));
        // T-438 item 14: the effect passes' CPU encoding time lands in FrameStats::effectsMs.
        CHECK(perf.current().effectsMs > 0.0f);
        CHECK(perf.current().renderMs >= perf.current().effectsMs);
    }
}

TEST_CASE("wgpu effects: a gradient mask and a mask node match SW") {
    CommandList cl = makeList(kS, kS);
    GradientStop stops[2] = {GradientStop{0, Color{0, 0, 0, 1}}, GradientStop{1, Color{0, 0, 0, 0}}};
    const uint32_t stopOffset = static_cast<uint32_t>(cl.stops.size());
    cl.stops.insert(cl.stops.end(), stops, stops + 2);
    EffectInstance gm = effect(EffectKind::GradientMask, {30, 30, 66, 66, 0, 0});
    gm.stopOffset = stopOffset;
    gm.stopCount = 2;
    addPushEffect(cl, {gm}, Rect{30, 30, 36, 36}, Rect{30, 30, 36, 36});
    addFillRect(cl, 30, 30, 36, 36, Color{0.2f, 1, 0.4f, 1});
    addPopEffect(cl);
    Session s(kS, kS);
    const std::vector<uint8_t> gpu = s.render(cl);
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(matches(gpu, renderSw(cl, kS, kS), kS, kS, "gradient mask"));
}

TEST_CASE("wgpu composite: the separable and non-separable blend modes match SW") {
    constexpr int kW = 256, kH = 64;
    CommandList cl = makeList(kW, kH);
    // A backdrop with a gradient of colours, then one layer per mode (Docs/design/09: 17 modes).
    for (int i = 0; i < 16; ++i) {
        addFillRect(cl, static_cast<float>(i) * 16.0f, 0, 16, kH,
                    Color{static_cast<float>(i) / 15.0f, 0.5f, 1.0f - static_cast<float>(i) / 15.0f, 1});
    }
    for (int mode = 1; mode <= 16; ++mode) {
        const float x = static_cast<float>(mode - 1) * 16.0f;
        addPushLayer(cl, 0.85f, static_cast<BlendMode>(mode), Rect{x, 8, 16, 48});
        addFillRect(cl, x + 2, 8, 12, 48, Color{0.9f, 0.3f, 0.6f, 0.8f});
        addPopLayer(cl);
    }
    Session s(kW, kH);
    const std::vector<uint8_t> gpu = s.render(cl);
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(matches(gpu, renderSw(cl, kW, kH), kW, kH, "blend modes"));
    // The non-Normal modes read a backdrop copy: copy passes ran.
    CHECK(s.renderer.stats().copies + s.renderer.stats().blits > 0u);
}

TEST_CASE("wgpu particles: DrawParticles (ui_particles.xt at 0.5 s) matches SW") {
    constexpr int kW = 256, kH = 256;
    const std::string dir = XT_GOLDEN_UI_DIR;
    std::ifstream in(dir + "/ui_particles.xt", std::ios::binary);
    REQUIRE(in.good());
    const std::string source((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    compile::CompileOptions options;
    options.emitMeta = false;
    options.rootDir = dir;
    std::vector<compile::SourceFile> files{{"ui_particles.xt", source}};
    compile::CompileResult result = compile::compileDocumentFromStrings(files, "ui_particles.xt", options);
    REQUIRE(result.ok());
    FileAssetResolver assets(dir, sw::decodeImageBytes);
    std::unique_ptr<Document> doc;
    REQUIRE(Document::load(IrBlob{result.ir.data(), result.ir.size()}, false, nullptr, &assets, doc) ==
            Status::kOk);
    doc->setViewport(Rect{0, 0, kW, kH});
    doc->update(0);
    doc->update(0.5f);
    CommandList list;
    doc->paint(list);
    list.viewport = Rect{0, 0, kW, kH};
    list.devicePixelRatio = 1;
    size_t instances = 0;
    for (const Command& c : list.commands) {
        if (c.kind == CmdKind::DrawParticles) instances += list.particles[c.index].count;
    }
    CHECK(instances > 40);
    Session s(kW, kH);
    const std::vector<uint8_t> gpu = s.render(list, Color{1, 1, 1, 1}, &assets, &doc->ir());
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(matches(gpu, renderSw(list, kW, kH, Color{1, 1, 1, 1}, &assets, &doc->ir()), kW, kH, "particles"));
}

namespace {

/// A host scene (gradient with stripes), straight = premultiplied (opaque).
std::vector<uint8_t> makeScene(int w, int h) {
    std::vector<uint8_t> s(static_cast<size_t>(w) * static_cast<size_t>(h) * 4);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uint8_t* p =
                s.data() + (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4u;
            p[0] = static_cast<uint8_t>(40 + (x * 180) / w);
            p[1] = static_cast<uint8_t>(((x / 12 + y / 12) % 2) != 0 ? 200 : 60);
            p[2] = static_cast<uint8_t>(60 + (y * 160) / h);
            p[3] = 255;
        }
    }
    return s;
}

/// Tests/gl/test_gl_direct.cpp's direct_glass_blend list, reduced: a band, two glass panels
/// (BackdropBlur scene runs), a multiply layer over everything, a final band.
CommandList glassList(int w, int h) {
    CommandList cl = makeList(w, h);
    cl.hostTargetMode = HostTargetMode::Direct;
    const float fw = static_cast<float>(w), fh = static_cast<float>(h);
    addFillRect(cl, 8, 8, fw * 0.3f, 14, Color{0, 0, 0, 0.5f});
    addFillRect(cl, 10, 10, fw * 0.2f, 10, Color{0.9f, 0.2f, 0.2f, 1});
    for (int i = 0; i < 2; ++i) {
        const Rect box{fw * (0.1f + 0.45f * static_cast<float>(i)), fh * 0.5f, fw * 0.35f, fh * 0.35f};
        addPushEffect(cl, {effect(EffectKind::BackdropBlur, {i == 0 ? 8.0f : 16.0f})}, box, box, true);
        addFillRect(cl, box.x, box.y, box.w, box.h, Color{1, 1, 1, 0.25f});
        addPopEffect(cl);
    }
    addPushLayer(cl, 1, BlendMode::Multiply, Rect{0, 0, fw, fh}, true);
    addFillRect(cl, 0, 0, fw, fh * 0.12f, Color{0.6f, 0.1f, 0.1f, 0.7f});
    addPopLayer(cl);
    addFillRect(cl, fw - 40, 8, 32, 12, Color{0.2f, 0.3f, 0.9f, 1});
    return cl;
}

}  // namespace

TEST_CASE("wgpu direct: T-343 Direct mode over a host scene matches SW, bands are pasted later") {
    constexpr int kW = 320, kH = 200;
    const std::vector<uint8_t> scene = makeScene(kW, kH);
    const CommandList cl = glassList(kW, kH);
    std::vector<uint8_t> swPixels = scene;
    {
        sw::SwRenderer sw;
        sw::SwTarget t{swPixels.data(), kW, kH, kW * 4};
        sw.setTarget(t);
        sw.render(cl, nullptr, nullptr);
    }
    Session s(kW, kH);
    s.renderer.disableClear();
    PerfTracker perf;
    perf.setEnabled(true);
    auto frame = [&](const DamageList* damage) {
        s.target.upload(scene);
        perf.beginFrame(0.0f);
        s.renderer.render(cl, nullptr, nullptr, s.target.view, nullptr, &perf, damage);
        CHECK_MESSAGE(std::string(s.renderer.lastError()).empty(), std::string(s.renderer.lastError()));
        return s.read();
    };
    const std::vector<uint8_t> first = frame(nullptr);
    CHECK(matches(first, swPixels, kW, kH, "direct glass"));
    CHECK(perf.current().bands > 0);
    CHECK(s.renderer.stats().cacheStores == perf.current().bands);
    // The scene survives where nothing was drawn (target 0 is never cleared).
    const size_t corner = (static_cast<size_t>(kH / 2) * kW + 4) * 4u;
    CHECK(first[corner] == scene[corner]);
    DamageList none;
    none.full = false;
    const std::vector<uint8_t> second = frame(&none);
    CHECK(perf.current().bandRedraws == 0);
    CHECK(s.renderer.stats().cacheStores == 0u);
    golden::GoldenResult same = golden::compareRgba(second.data(), first.data(), kW, kH, 1, 0.0f);
    CHECK(same.pass);
}

TEST_CASE("wgpu effects: a repeated effect segment is stored, then drawn from its cached target (T-162)") {
    CommandList cl = effectList({effect(EffectKind::DropShadow, {6, 6, 4, 0, 1})});
    Session s(kS, kS);
    const std::vector<uint8_t> first = s.render(cl);
    CHECK(s.renderer.stats().cacheStores == 0u);
    CHECK(s.renderer.stats().fxCacheMisses == 1u);  // T-658: the builder's counters
    CHECK(s.renderer.stats().fxCacheStores == 0u);
    const std::vector<uint8_t> second = s.render(cl);
    CHECK(s.renderer.stats().cacheStores == 1u);
    CHECK(s.renderer.stats().fxCacheMisses == 1u);
    CHECK(s.renderer.stats().fxCacheStores == 1u);
    const std::vector<uint8_t> third = s.render(cl);
    CHECK(s.renderer.stats().cacheStores == 0u);
    CHECK(s.renderer.stats().fxCacheHits == 1u);
    CHECK(s.renderer.stats().fxCacheMisses == 0u);
    CHECK_MESSAGE(s.error.empty(), s.error);
    golden::GoldenResult r2 = golden::compareRgba(second.data(), first.data(), kS, kS, 0, 0.0f);
    golden::GoldenResult r3 = golden::compareRgba(third.data(), first.data(), kS, kS, 1, 0.0f);
    CHECK(r2.pass);
    CHECK(r3.pass);
}

TEST_CASE("wgpu custom: XtInput.uv is top-down, parameters reach the block, output is premultiplied") {
    const std::string glsl = glslOf(
        "param k float 1\nparam tint color #ffffff\n---\n"
        "float4 xtMain(XtInput i) {\n"
        "    float4 c = xtSample(i.uv);\n"
        "    return float4(i.uv.x, i.uv.y * k, tint.b, c.a);\n"
        "}\n");
    auto listWith = [&](float k, float alpha) {
        CommandList cl = makeList(64, 64);
        addShader(cl, 3, glsl, {"k", "tint"}, {0, 2}, EffectInstance{});
        addPushEffect(cl, {customFx(3, {k}, Color{0, 0, 1, 1})}, Rect{0, 0, 64, 64}, Rect{0, 0, 64, 64});
        addFillRect(cl, 0, 0, 64, 64, Color{1, 1, 1, alpha});
        addPopEffect(cl);
        return cl;
    };
    Session s(64, 64);
    const std::vector<uint8_t> a = s.render(listWith(1.0f, 1.0f));
    CHECK_MESSAGE(s.warning.empty(), s.warning);
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(near(pixel(a, 64, 8, 8)[0], 34) <= 6);
    CHECK(near(pixel(a, 64, 8, 8)[1], 34) <= 6);
    CHECK(near(pixel(a, 64, 8, 56)[1], 227) <= 6);
    CHECK(near(pixel(a, 64, 56, 8)[0], 227) <= 6);
    CHECK(pixel(a, 64, 8, 8)[2] > 245);
    const std::vector<uint8_t> b = s.render(listWith(0.5f, 1.0f));
    CHECK(near(pixel(b, 64, 8, 56)[1], 113) <= 6);
    const std::vector<uint8_t> c = s.render(listWith(1.0f, 0.5f));
    CHECK(near(pixel(c, 64, 8, 56)[1], 113) <= 8);
}

TEST_CASE("wgpu custom: a shader that fails to compile warns once and applies its fallback") {
    CommandList cl = makeList(64, 64);
    addShader(cl, 1, "this is not GLSL;", {}, {}, effect(EffectKind::Blur, {4, 4}));
    addPushEffect(cl, {customFx(1, {})}, Rect{0, 0, 64, 64}, Rect{16, 16, 32, 32});
    addFillRect(cl, 16, 16, 32, 32, Color{1, 0.5f, 0, 1});
    addPopEffect(cl);
    Session s(64, 64);
    const std::vector<uint8_t> first = s.render(cl);
    CHECK(s.warning.find("failed to compile") != std::string::npos);
    CHECK_MESSAGE(s.error.empty(), s.error);
    const std::vector<uint8_t> second = s.render(cl);
    CHECK(s.warning.empty());  // reported once
    CHECK(matches(second, renderSw(cl, 64, 64), 64, 64, "custom fallback"));
    golden::GoldenResult same = golden::compareRgba(first.data(), second.data(), 64, 64, 0, 0.0f);
    CHECK(same.pass);
}

// T-892 (18 §18.7, 59 §59.4): the wgpu wrapper only adds listed `xt` names (xt/xtfx_reserved.h), and
// a module the GLSL frontend rejects reports the first line of its message in the effect report.

TEST_CASE("wgpu custom: T-892 every xt name of the wrapper is in the reserved list") {
    const std::string module = glslOf(
        "param k float 1\nparam tint color #fff\nparam off vec2 0 0\n---\n"
        "float4 xtMain(XtInput i) { return xtSample(i.uv + off + fmod(i.local, i.boxSize)) * tint * k * "
        "xtTime; }\n");
    const char names[] = "k\0tint\0off\0";
    const uint8_t types[] = {0, 2, 1};
    const std::string wrapped = xt::wgpu::detail::wrapCustomModule(module, names, types, 3);
    const std::vector<std::string> unlisted = xtfxUnlistedWrapperNames(wrapped);
    CHECK_MESSAGE(unlisted.empty(), (unlisted.empty() ? std::string() : unlisted.front()) << "\n" << wrapped);
    // The three names T-889 found: the module compiled with them as parameters and fell back.
    for (const char* name : {"xtU", "xtLayerTex", "xtLayerSmp"}) {
        CHECK(wrapped.find(name) != std::string::npos);
        CHECK(isXtfxWrapperName(name));
    }
}

TEST_CASE("wgpu custom: T-892 a shader the GLSL frontend rejects reports its message in the effect report") {
    // A valid body's translation with a vec4 assigned to a vec3 put in (a bad translation result).
    std::string glsl = glslOf("---\nfloat4 xtMain(XtInput i) { return xtSample(i.uv); }\n");
    const std::string ret = "    return xtSample(i.uv);\n";
    const size_t at = glsl.find(ret);
    REQUIRE(at != std::string::npos);
    glsl.insert(at, "    vec3 bad = vec4(1.0, 1.0, 1.0, 1.0);\n");
    CommandList cl = makeList(64, 64);
    addShader(cl, 1, glsl, {}, {}, effect(EffectKind::Blur, {4, 4}));
    addPushEffect(cl, {customFx(1, {})}, Rect{0, 0, 64, 64}, Rect{16, 16, 32, 32});
    addFillRect(cl, 16, 16, 32, 32, Color{1, 0.5f, 0, 1});
    addPopEffect(cl);
    Session s(64, 64);
    std::vector<gpu::FrameBuilder::EffectReport> reports;
    s.renderer.setEffectReport(&reports);
    s.render(cl);
    REQUIRE(reports.size() == 1);
    CHECK_FALSE(reports[0].executed);
    CHECK(reports[0].drawnInstead == EffectKind::Blur);
    const std::string failure = reports[0].failure;
    REQUIRE_FALSE(failure.empty());
    CHECK(s.warning.find(failure.substr(0, failure.find('\n'))) != std::string::npos);
    const std::string reason = gpu::FrameBuilder::effectFailureReason("wgpu", failure);
    MESSAGE("wgpu reason: " << reason);
    CHECK(reason.rfind("shader failed to compile on wgpu: ", 0) == 0);
    CHECK(reason.find('\n') == std::string::npos);
    // Cached with its message: the next frame reports the same text and warns no more.
    s.render(cl);
    CHECK(s.warning.empty());
    REQUIRE(reports.size() == 1);
    CHECK(reports[0].failure == failure);
    s.renderer.setEffectReport(nullptr);
}

// T-818 (Docs/design/18-effects.md §18.7.x): XtInput.local follows the node's transform, XtInput.uv
// stays in screen space, XtInput.boxSize is the border box in CSS px (the cases of
// Tests/gl/test_gl_effects.cpp through xt_wgpu).

namespace {

/// A node with border box (0, 0, `box.x`, `box.y`) drawn through `xf` under a custom effect
/// (`asset`) whose window is the transformed box's AABB clipped to the viewport.
CommandList localNodeList(int size, AssetId asset, const std::string& glsl, Mat2x3 xf, Vec2 box) {
    CommandList cl = makeList(size, size);
    addShader(cl, asset, glsl, {}, {}, EffectInstance{});
    Rect aabb = transformBounds(xf, Rect{0, 0, box.x, box.y});
    aabb = intersect(aabb, cl.viewport);
    addPushEffect(cl, {customFx(asset, {})}, aabb, aabb);
    cl.effectCmds.back().xf = xf;
    cl.effectCmds.back().localBox = Rect{0, 0, box.x, box.y};
    addFillRect(cl, 0, 0, box.x, box.y, Color{1, 1, 1, 1});
    cl.fills.back().xf = xf;
    addPopEffect(cl);
    return cl;
}

}  // namespace

TEST_CASE("wgpu custom: XtInput.local follows translation, scale and rotation (T-818)") {
    const std::string glsl = glslOf(
        "---\n"
        "float4 xtMain(XtInput i) {\n"
        "    return float4(i.local.x, i.local.y, i.boxSize.x / 64, 1);\n"
        "}\n");
    Session s(64, 64);
    // Each transform puts the node's local point (4.5, 12.5) -- (0.28125, 0.78125) in `local` --
    // on the center of pixel (x, y).
    struct Case {
        const char* what;
        Mat2x3 xf;
        int x, y;
    };
    const Case cases[] = {
        {"translate (8, 8)", translation(Vec2{8, 8}), 12, 20},
        {"translate (30, 20)", translation(Vec2{30, 20}), 34, 32},
        {"scale 2", mul(translation(Vec2{10.5f, 10.5f}), scaling(Vec2{2, 2})), 19, 35},
        {"rotate 90", mul(translation(Vec2{40, 8}), rotationDeg(90)), 27, 12},
    };
    for (const Case& c : cases) {
        INFO(c.what);
        const std::vector<uint8_t> a = s.render(localNodeList(64, 5, glsl, c.xf, Vec2{16, 16}));
        CHECK_MESSAGE(s.warning.empty(), s.warning);
        CHECK_MESSAGE(s.error.empty(), s.error);
        CHECK(near(pixel(a, 64, c.x, c.y)[0], 72) <= 2);   // local.x = 4.5 / 16
        CHECK(near(pixel(a, 64, c.x, c.y)[1], 199) <= 2);  // local.y = 12.5 / 16
        CHECK(near(pixel(a, 64, c.x, c.y)[2], 64) <= 2);   // boxSize.x / 64
    }
}

TEST_CASE("wgpu custom: a full-screen node moved: local follows it, uv does not (T-818)") {
    const std::string glsl = glslOf(
        "---\n"
        "float4 xtMain(XtInput i) {\n"
        "    return float4(i.uv.x, i.local.x, 0, 1);\n"
        "}\n");
    Session s(64, 64);
    const std::vector<uint8_t> a =
        s.render(localNodeList(64, 6, glsl, translation(Vec2{0, 0}), Vec2{64, 64}));
    const std::vector<uint8_t> b =
        s.render(localNodeList(64, 6, glsl, translation(Vec2{8, 0}), Vec2{64, 64}));
    CHECK_MESSAGE(s.error.empty(), s.error);
    CHECK(near(pixel(a, 64, 32, 32)[0], 130) <= 2);  // uv.x = 32.5 / 64
    CHECK(near(pixel(b, 64, 32, 32)[0], 130) <= 2);  // the window (the viewport) did not move
    CHECK(near(pixel(a, 64, 32, 32)[1], 130) <= 2);  // local.x = 32.5 / 64
    CHECK(near(pixel(b, 64, 32, 32)[1], 98) <= 2);   // local.x = 24.5 / 64
}

// T-831: a scene node's geometry box does not start at its local origin; `local` is
// (local point - box.xy) / box.wh.
TEST_CASE("wgpu custom: XtInput.local over a box whose origin is not 0,0 (T-831)") {
    const std::string glsl = glslOf(
        "---\n"
        "float4 xtMain(XtInput i) {\n"
        "    return float4(i.local.x, i.local.y, i.boxSize.x / 64, 1);\n"
        "}\n");
    CommandList cl = makeList(64, 64);
    addShader(cl, 8, glsl, {}, {}, EffectInstance{});
    const Mat2x3 xf = translation(Vec2{10, 10});
    const Rect box{20, 8, 16, 16};
    const Rect aabb = transformBounds(xf, box);
    addPushEffect(cl, {customFx(8, {})}, aabb, aabb);
    cl.effectCmds.back().xf = xf;
    cl.effectCmds.back().localBox = box;
    addFillRect(cl, box.x, box.y, box.w, box.h, Color{1, 1, 1, 1});
    cl.fills.back().xf = xf;
    addPopEffect(cl);
    Session s(64, 64);
    const std::vector<uint8_t> a = s.render(cl);
    CHECK_MESSAGE(s.warning.empty(), s.warning);
    CHECK_MESSAGE(s.error.empty(), s.error);
    // Local point (24.5, 20.5) = box point (4.5, 12.5) sits at screen pixel (34, 30).
    CHECK(near(pixel(a, 64, 34, 30)[0], 72) <= 2);   // (24.5 - 20) / 16
    CHECK(near(pixel(a, 64, 34, 30)[1], 199) <= 2);  // (20.5 - 8) / 16
    CHECK(near(pixel(a, 64, 34, 30)[2], 64) <= 2);   // boxSize.x / 64
}

TEST_CASE("wgpu custom: XtInput.boxSize is the border box in CSS px at any dpr (T-818)") {
    const std::string glsl = glslOf(
        "---\n"
        "float4 xtMain(XtInput i) {\n"
        "    return float4(i.boxSize.x / 64, i.boxSize.y / 64, i.local.x, 1);\n"
        "}\n");
    for (int dpr : {1, 2}) {
        INFO("dpr " << dpr);
        const int size = 64 * dpr;
        CommandList cl = localNodeList(64, 7, glsl, translation(Vec2{8, 8}), Vec2{16, 32});
        cl.devicePixelRatio = static_cast<float>(dpr);
        Session s(size, size);
        const std::vector<uint8_t> a = s.render(cl);
        CHECK_MESSAGE(s.error.empty(), s.error);
        // The device pixel whose center is the local point (4.25, 8.25) at dpr 2, (4.5, 8.5) at dpr 1.
        const int x = dpr == 2 ? 24 : 12;
        const int y = dpr == 2 ? 32 : 16;
        CHECK(near(pixel(a, size, x, y)[0], 64) <= 2);                  // 16 / 64
        CHECK(near(pixel(a, size, x, y)[1], 128) <= 2);                 // 32 / 64
        CHECK(near(pixel(a, size, x, y)[2], dpr == 2 ? 68 : 72) <= 2);  // local.x
    }
}
