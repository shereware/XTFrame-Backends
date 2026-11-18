// T-729 (Docs/design/10-backends.md §10.18 "B 定案", Docs/tasks/T-729.md "測試"): the cases of
// Tests/wgpu/test_wgpu_backdrop_mask.cpp on xt_gl. Each command list is built against the renderer's
// GL device seen through a forwarder that reports supportsBackdropMask() or not, and both frames are
// drawn by that device into the same FBO: a stack of BackdropBlur followed only by shadow-type effects
// takes the new semantics (one BackdropMasked pass, no extra pass) within 15 §15.3's tolerance of
// xt::sw; every other stack is byte-identical either way.

#include <SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "golden_common.h"
#include "xt/command_list.h"
#include "xt/gl/gl_renderer.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_device.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/sw/sw_renderer.h"
#include "xt/types.h"

using namespace xt;

namespace {

constexpr int kW = 256, kH = 192;

void* sdlLoadProcMask(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

/// The GL device as the frame builder sees it, with the backdrop mask reported as asked.
class MaskForwarder : public gpu::GpuDevice {
public:
    MaskForwarder(gpu::GpuDevice& inner, bool mask) : m_inner(inner), m_mask(mask) {}
    gpu::TextureHandle createTexture(const char* name, uint16_t w, uint16_t h, gpu::TextureFormat fmt,
                                     const void* initial) override {
        return m_inner.createTexture(name, w, h, fmt, initial);
    }
    void updateTexture(gpu::TextureHandle tex, uint16_t x, uint16_t y, uint16_t w, uint16_t h,
                       const void* rgba, size_t rowPitch) override {
        m_inner.updateTexture(tex, x, y, w, h, rgba, rowPitch);
    }
    void destroyTexture(gpu::TextureHandle tex) override { m_inner.destroyTexture(tex); }
    void submit(const gpu::GpuFrame& frame) override { m_inner.submit(frame); }
    gpu::TextureHandle acquireCachedTarget(uint16_t w, uint16_t h, gpu::TextureFormat fmt) override {
        return m_inner.acquireCachedTarget(w, h, fmt);
    }
    void releaseCachedTarget(gpu::TextureHandle target) override { m_inner.releaseCachedTarget(target); }
    uint32_t maxTextureSize() const override { return m_inner.maxTextureSize(); }
    bool supportsCoverageDraws() const override { return m_inner.supportsCoverageDraws(); }
    bool supportsBoxShadowShading() const override { return m_inner.supportsBoxShadowShading(); }
    bool supportsAnalyticClip() const override { return m_inner.supportsAnalyticClip(); }
    bool supportsAlphaChannelReads() const override { return m_inner.supportsAlphaChannelReads(); }
    bool supportsBackdropMask() const override { return m_mask; }

private:
    gpu::GpuDevice& m_inner;
    bool m_mask;
};

struct Drawn {
    std::vector<uint8_t> px;  ///< top-down
    size_t passes = 0;
    uint32_t masked = 0;
};

/// A GlRenderer and its own single-sample RGBA8 + stencil FBO.
struct Session {
    gl::GLuint fbo = 0, color = 0, rbo = 0;
    gl::GlRenderer renderer;

    Session() {
        REQUIRE(renderer.init(sdlLoadProcMask));
        using namespace xt::gl;
        glGenTextures(1, &color);
        glBindTexture(GL_TEXTURE_2D, color);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8), kW, kH, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                     nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_NEAREST));
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, static_cast<GLint>(GL_NEAREST));
        glGenRenderbuffers(1, &rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, kW, kH);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
        REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        renderer.resize(kW, kH);
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

    /// Builds `cl` for the device as `mask` reports it, clears to opaque black, draws, reads.
    Drawn draw(const CommandList& cl, bool mask) {
        using namespace xt::gl;
        gpu::GpuDevice* device = renderer.device();
        REQUIRE(device != nullptr);
        MaskForwarder forwarder(*device, mask);
        gpu::FrameBuilder builder;
        builder.init(forwarder);
        builder.resize(kW, kH);
        builder.setEdgeFringe(true);
        gpu::GpuFrame frame;
        builder.build(cl, nullptr, nullptr, nullptr, frame);
        Drawn out;
        out.passes = frame.passes.size();
        for (const gpu::DrawCmd& c : frame.cmds) {
            out.masked += c.shading == gpu::Shading::Effect && c.program == gpu::EffectProgram::BackdropMasked
                              ? 1u
                              : 0u;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glViewport(0, 0, kW, kH);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilMask(0xFFu);
        glClearColor(0, 0, 0, 1);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        renderer.submitFrame(frame);
        CHECK_MESSAGE(std::string(renderer.lastError()).empty(), renderer.lastError());
        std::vector<uint8_t> px(static_cast<size_t>(kW) * kH * 4);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, kW, kH, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        builder.shutdown();
        out.px.resize(px.size());
        const size_t row = static_cast<size_t>(kW) * 4;
        for (int y = 0; y < kH; ++y) {
            std::copy(px.begin() + static_cast<long>(static_cast<size_t>(kH - 1 - y) * row),
                      px.begin() + static_cast<long>(static_cast<size_t>(kH - y) * row),
                      out.px.begin() + static_cast<long>(static_cast<size_t>(y) * row));
        }
        return out;
    }
};

PathRef polygon(CommandList& cl, std::initializer_list<Vec2> pts) {
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

void fill(CommandList& cl, PathRef path, Color color) {
    FillCmd f{};
    f.path = path;
    f.paint.kind = PaintKind::Solid;
    f.paint.color = color;
    f.rule = FillRule::NonZero;
    f.antialias = 1;
    cl.fills.push_back(f);
    cl.commands.push_back(Command{CmdKind::FillPath, {}, static_cast<uint32_t>(cl.fills.size() - 1)});
}

void fillRect(CommandList& cl, float x, float y, float w, float h, Color color) {
    fill(cl, polygon(cl, {Vec2{x, y}, Vec2{x + w, y}, Vec2{x + w, y + h}, Vec2{x, y + h}}), color);
}

EffectInstance fx(EffectKind kind, std::initializer_list<float> params, Color color = Color{0, 0, 0, 1}) {
    EffectInstance e{};
    e.kind = kind;
    int i = 0;
    for (float p : params) {
        e.params[i++] = p;
    }
    e.colors[0] = color;
    e.enabled = 1;
    return e;
}
EffectInstance blurFx() { return fx(EffectKind::BackdropBlur, {10, 0.4f, 0.1f}); }
EffectInstance outerFx() {
    return fx(EffectKind::DropShadow, {3, 7, 12, 0, 1, 0}, Color{0.05f, 0, 0.1f, 0.75f});
}
EffectInstance insetFx() { return fx(EffectKind::DropShadow, {0, 3, 8, 0, 1, 1}, Color{0, 0, 0.2f, 0.6f}); }
EffectInstance glowFx(bool inner) {
    return fx(EffectKind::Glow, {10, 1.3f, inner ? 1.0f : 0.0f}, Color{1, 0.85f, 0.2f, 1});
}
EffectInstance adjustFx() { return fx(EffectKind::ColorAdjust, {0.1f, 0, -0.4f, 0, 1}); }

/// Tests/wgpu/test_wgpu_backdrop_mask.cpp's glassList.
CommandList glassList(std::initializer_list<EffectInstance> effects) {
    CommandList cl;
    cl.viewport = Rect{0, 0, kW, kH};
    cl.devicePixelRatio = 1;
    for (int i = 0; i < 8; ++i) {
        fillRect(cl, static_cast<float>(i) * 32.0f, 0, 32, kH,
                 Color{0.12f * static_cast<float>(i), 0.35f, 0.9f - 0.1f * static_cast<float>(i), 1});
    }
    const Rect panel{64, 48, 128, 80};
    EffectCmd e{};
    e.effectOffset = static_cast<uint32_t>(cl.effects.size());
    for (const EffectInstance& f : effects) {
        cl.effects.push_back(f);
    }
    e.effectCount = static_cast<uint32_t>(effects.size());
    e.xf = Mat2x3{};
    e.boundsHint = Rect{34, 18, 188, 140};
    e.contentBounds = panel;
    cl.effectCmds.push_back(e);
    cl.commands.push_back(Command{CmdKind::PushEffect, {}, static_cast<uint32_t>(cl.effectCmds.size() - 1)});
    fillRect(cl, panel.x, panel.y, panel.w, panel.h, Color{1, 1, 1, 0.5f});
    fill(cl, polygon(cl, {Vec2{90.3f, 60.6f}, Vec2{170.2f, 70.1f}, Vec2{120.7f, 118.4f}}),
         Color{0.2f, 0.9f, 0.6f, 0.65f});
    cl.commands.push_back(Command{CmdKind::PopEffect, {}, 0});
    return cl;
}

std::vector<uint8_t> renderSw(const CommandList& cl) {
    std::vector<uint8_t> buf(static_cast<size_t>(kW) * kH * 4, 0);
    sw::SwRenderer renderer;
    sw::SwTarget target{buf.data(), kW, kH, kW * 4};
    renderer.setTarget(target);
    renderer.clear(Color{0, 0, 0, 1});
    renderer.render(cl, nullptr, nullptr);
    return buf;
}

size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
        n += (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2] || a[i + 3] != b[i + 3]) ? 1u : 0u;
    }
    return n;
}

bool matchesSw(const std::vector<uint8_t>& gl, const std::vector<uint8_t>& sw, const char* what) {
    golden::GoldenResult r = golden::compareRgba(gl.data(), sw.data(), kW, kH, 16, 0.03f);
    std::printf("gl vs sw %s: %u px differ, max channel diff %u\n", what, r.diffPixels, r.maxChannelDiff);
    return r.pass;
}

void checkNewSemantics(Session& s, const char* what, std::initializer_list<EffectInstance> effects) {
    CAPTURE(what);
    const CommandList cl = glassList(effects);
    const Drawn masked = s.draw(cl, true);
    const Drawn legacy = s.draw(cl, false);
    CHECK(masked.masked == 1u);
    CHECK(legacy.masked == 0u);
    CHECK(masked.passes == legacy.passes);
    const std::vector<uint8_t> sw = renderSw(cl);
    CHECK(matchesSw(masked.px, sw, what));
    golden::GoldenResult old = golden::compareRgba(legacy.px.data(), sw.data(), kW, kH, 16, 1.0f);
    std::printf("gl backdrop mask %s: legacy semantics vs sw %u px differ, max %u; %zu px changed\n", what,
                old.diffPixels, old.maxChannelDiff, differing(masked.px, legacy.px));
}

void checkUnchanged(Session& s, const char* what, std::initializer_list<EffectInstance> effects) {
    CAPTURE(what);
    const CommandList cl = glassList(effects);
    const Drawn masked = s.draw(cl, true);
    const Drawn legacy = s.draw(cl, false);
    CHECK(masked.masked == 0u);
    CHECK(masked.passes == legacy.passes);
    CHECK(differing(masked.px, legacy.px) == 0u);
    CHECK(matchesSw(masked.px, renderSw(cl), what));
}

}  // namespace

TEST_CASE("gl backdrop mask: shadows under a glass panel follow the new semantics and match SW (T-729)") {
    Session s;
    checkNewSemantics(s, "outer shadow", {blurFx(), outerFx()});
    checkNewSemantics(s, "inset shadow", {blurFx(), insetFx()});
    checkNewSemantics(s, "glow", {blurFx(), glowFx(false)});
    checkNewSemantics(s, "inner glow", {blurFx(), glowFx(true)});
    checkNewSemantics(s, "outer and inset", {blurFx(), outerFx(), insetFx()});
}

TEST_CASE("gl backdrop mask: other stacks are byte-identical with and without the capability (T-729)") {
    Session s;
    checkUnchanged(s, "plain backdrop blur", {blurFx()});
    checkUnchanged(s, "backdrop blur then color adjust", {blurFx(), adjustFx()});
    checkUnchanged(s, "backdrop blur, shadow, color adjust", {blurFx(), outerFx(), adjustFx()});
    checkUnchanged(s, "backdrop blur not first", {outerFx(), blurFx()});
}
