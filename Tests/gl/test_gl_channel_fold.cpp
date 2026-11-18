// T-728 (Docs/design/10-backends.md §10.18 A item 1, Docs/tasks/T-728.md "測試"): xt_gl draws the
// folded chains (the AlphaExtract pass folded into the next Downsample / GaussScalar pass through
// GaussParams / DownsampleParams `channel`) byte for byte like the unfolded ones. The cases of
// Tests/wgpu/test_wgpu_channel_fold.cpp: each command list is built twice, against the renderer's GL
// device seen through a forwarder that reports supportsAlphaChannelReads() or not, and both frames
// are drawn by that device into the same FBO. Also item 3 (§10.18 A item 3): a glass panel on an
// empty layer, drawn without its pyramid, is byte-identical to the same list with a fully transparent
// fill under the panel first (which keeps the all-zero pyramid). T-735: inset shadows and inner glows
// (`1 - .a`) are not folded -- same passes, same bytes.

#include <SDL.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "xt/command_list.h"
#include "xt/gl/gl_renderer.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_device.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/types.h"

using namespace xt;

namespace {

constexpr int kW = 256, kH = 192;

void* sdlLoadProcFold(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

/// The GL device as the frame builder sees it, with the channel reads reported as asked.
class ChannelForwarder : public gpu::GpuDevice {
public:
    ChannelForwarder(gpu::GpuDevice& inner, bool channels) : m_inner(inner), m_channels(channels) {}
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
    uint32_t prepareCustomEffect(std::string_view glsl, const char* names, const uint8_t* types,
                                 uint32_t count) override {
        return m_inner.prepareCustomEffect(glsl, names, types, count);
    }
    uint32_t maxTextureSize() const override { return m_inner.maxTextureSize(); }
    bool supportsCoverageDraws() const override { return m_inner.supportsCoverageDraws(); }
    bool supportsBoxShadowShading() const override { return m_inner.supportsBoxShadowShading(); }
    bool supportsAnalyticClip() const override { return m_inner.supportsAnalyticClip(); }
    bool supportsAlphaChannelReads() const override { return m_channels; }

private:
    gpu::GpuDevice& m_inner;
    bool m_channels;
};

/// A GlRenderer and its own single-sample RGBA8 + stencil FBO.
struct Session {
    gl::GLuint fbo = 0, color = 0, rbo = 0;
    gl::GlRenderer renderer;

    Session() {
        REQUIRE(renderer.init(sdlLoadProcFold));
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

    /// Builds `cl` for the device as `channels` reports it, clears to opaque black, draws, reads
    /// (rows bottom-up, as GL gives them: only compared with each other).
    std::vector<uint8_t> draw(const CommandList& cl, bool channels, size_t& passes,
                              uint32_t* copies = nullptr) {
        using namespace xt::gl;
        gpu::GpuDevice* device = renderer.device();
        REQUIRE(device != nullptr);
        ChannelForwarder forwarder(*device, channels);
        gpu::FrameBuilder builder;
        builder.init(forwarder);
        builder.resize(kW, kH);
        builder.setEdgeFringe(true);
        gpu::GpuFrame frame;
        builder.build(cl, nullptr, nullptr, nullptr, frame);
        passes = frame.passes.size();
        if (copies != nullptr) {
            *copies = 0;
            for (const gpu::Pass& p : frame.passes) {
                *copies += p.copy != 0 ? 1u : 0u;
            }
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
        return px;
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

/// Tests/wgpu/test_wgpu_channel_fold.cpp's foldList: every alpha value, a translucent 7-gon and
/// scattered translucent triangles inside one effect group.
CommandList foldList(std::initializer_list<EffectInstance> effects) {
    CommandList cl;
    cl.viewport = Rect{0, 0, kW, kH};
    cl.devicePixelRatio = 1;
    fillRect(cl, 0, 0, kW, kH, Color{0.2f, 0.35f, 0.5f, 1});
    EffectCmd e{};
    e.effectOffset = static_cast<uint32_t>(cl.effects.size());
    e.effectCount = static_cast<uint32_t>(effects.size());
    for (const EffectInstance& fx : effects) {
        cl.effects.push_back(fx);
    }
    e.xf = Mat2x3{};
    e.boundsHint = Rect{0, 0, kW, kH};
    e.contentBounds = Rect{40, 36, 176, 120};
    cl.effectCmds.push_back(e);
    cl.commands.push_back(Command{CmdKind::PushEffect, {}, static_cast<uint32_t>(cl.effectCmds.size() - 1)});
    for (int i = 0; i < 128; ++i) {
        const float x = 64.0f + static_cast<float>(i);
        fillRect(cl, x, 44, 1, 22,
                 Color{static_cast<float>(i) / 127.0f, 0.4f, 0.9f, static_cast<float>(i) / 255.0f});
        fillRect(cl, x, 70, 1, 22,
                 Color{0.9f, static_cast<float>(i) / 127.0f, 0.3f, static_cast<float>(128 + i) / 255.0f});
    }
    PathRef gon{};
    gon.verbOffset = static_cast<uint32_t>(cl.verbs.size());
    gon.pointOffset = static_cast<uint32_t>(cl.points.size());
    for (int i = 0; i < 7; ++i) {
        const float a = 6.2831853f * static_cast<float>(i) / 7.0f;
        cl.verbs.push_back(i == 0 ? PathVerb::Move : PathVerb::Line);
        cl.points.push_back(96.3f + 23.6f * std::cos(a));
        cl.points.push_back(124.7f + 23.6f * std::sin(a));
    }
    cl.verbs.push_back(PathVerb::Close);
    gon.verbCount = static_cast<uint32_t>(cl.verbs.size()) - gon.verbOffset;
    gon.pointCount = static_cast<uint32_t>(cl.points.size()) - gon.pointOffset;
    fill(cl, gon, Color{1, 1, 1, 0.73f});
    uint32_t seed = 12345u;
    auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return static_cast<float>(seed >> 8) / 16777216.0f;
    };
    for (int t = 0; t < 24; ++t) {
        const float cx = 130.0f + next() * 80.0f;
        const float cy = 100.0f + next() * 50.0f;
        fill(cl,
             polygon(cl, {Vec2{cx + next() * 14.0f, cy - next() * 13.0f},
                          Vec2{cx - next() * 15.0f, cy + next() * 3.0f},
                          Vec2{cx + next() * 6.0f, cy + next() * 16.0f}}),
             Color{next(), next(), next(), 0.15f + next() * 0.85f});
    }
    cl.commands.push_back(Command{CmdKind::PopEffect, {}, 0});
    return cl;
}

EffectInstance dropShadow(float blur, float spread, bool inset) {
    EffectInstance e{};
    e.kind = EffectKind::DropShadow;
    e.params[0] = 4;
    e.params[1] = 5;
    e.params[2] = blur;
    e.params[3] = spread;
    e.params[4] = 1;
    e.params[5] = inset ? 1.0f : 0.0f;
    e.colors[0] = Color{0.05f, 0, 0.1f, 0.85f};
    e.enabled = 1;
    return e;
}

EffectInstance glow(float radius, bool inner) {
    EffectInstance e{};
    e.kind = EffectKind::Glow;
    e.params[0] = radius;
    e.params[1] = 1.4f;
    e.params[2] = inner ? 1.0f : 0.0f;
    e.colors[0] = Color{1, 0.85f, 0.2f, 1};
    e.enabled = 1;
    return e;
}

void checkSame(Session& s, const char* what, std::initializer_list<EffectInstance> effects, size_t folded) {
    CAPTURE(what);
    const CommandList cl = foldList(effects);
    size_t passesBefore = 0, passesAfter = 0;
    const std::vector<uint8_t> before = s.draw(cl, false, passesBefore);
    const std::vector<uint8_t> after = s.draw(cl, true, passesAfter);
    CHECK(passesAfter + folded == passesBefore);
    size_t differ = 0;
    for (size_t i = 0; i < after.size(); i += 4) {
        differ += std::equal(after.begin() + static_cast<long>(i), after.begin() + static_cast<long>(i + 4),
                             before.begin() + static_cast<long>(i))
                      ? 0u
                      : 1u;
    }
    std::printf("gl channel fold %s: %zu -> %zu passes, %zu px differ\n", what, passesBefore, passesAfter,
                differ);
    CHECK(differ == 0);
}

/// Tests/wgpu/test_wgpu_fewer_passes.cpp's glassOnLayerList (on this file's 256 x 192 surface).
CommandList glassOnLayerList(bool under, float underAlpha) {
    CommandList cl;
    cl.viewport = Rect{0, 0, kW, kH};
    cl.devicePixelRatio = 1;
    for (int i = 0; i < 8; ++i) {
        fillRect(cl, static_cast<float>(i) * 32.0f, 0, 32, kH,
                 Color{0.1f * static_cast<float>(i), 0.3f, 0.9f - 0.1f * static_cast<float>(i), 1});
    }
    LayerCmd layer{};
    layer.opacity = 0.9f;
    layer.blend = BlendMode::Normal;
    layer.boundsHint = Rect{0, 0, kW, kH};
    cl.layers.push_back(layer);
    cl.commands.push_back(Command{CmdKind::PushLayer, {}, 0});
    if (under) {
        fillRect(cl, 40, 40, 170, 110, Color{0.9f, 0.8f, 0.1f, underAlpha});
    }
    const Rect panel{60, 56, 130, 76};
    EffectInstance blur{};
    blur.kind = EffectKind::BackdropBlur;
    blur.params[0] = 10;
    blur.params[1] = 0.4f;
    blur.params[2] = 0.1f;
    blur.enabled = 1;
    EffectInstance outer = dropShadow(10, 0, false);
    EffectInstance inset = dropShadow(6, 0, true);
    EffectCmd e{};
    e.effectOffset = static_cast<uint32_t>(cl.effects.size());
    for (const EffectInstance& fx : {blur, outer, inset}) {
        cl.effects.push_back(fx);
    }
    e.effectCount = 3;
    e.xf = Mat2x3{};
    e.boundsHint = panel;
    e.contentBounds = panel;
    cl.effectCmds.push_back(e);
    cl.commands.push_back(Command{CmdKind::PushEffect, {}, static_cast<uint32_t>(cl.effectCmds.size() - 1)});
    fillRect(cl, panel.x, panel.y, panel.w, panel.h, Color{1, 1, 1, 0.22f});
    fill(cl, polygon(cl, {Vec2{90.3f, 70.6f}, Vec2{170.2f, 80.1f}, Vec2{120.7f, 125.4f}}),
         Color{0.2f, 0.9f, 0.6f, 0.65f});
    cl.commands.push_back(Command{CmdKind::PopEffect, {}, 0});
    fillRect(cl, 200, 140, 30, 30, Color{1, 0.2f, 0.2f, 1});  // after the panel
    cl.commands.push_back(Command{CmdKind::PopLayer, {}, 0});
    return cl;
}

}  // namespace

TEST_CASE(
    "gl fewer passes: a glass panel on an empty layer is byte-identical without its pyramid (T-728 item 3)") {
    Session s;
    size_t passesShortcut = 0, passesPyramid = 0;
    uint32_t copiesShortcut = 0, copiesPyramid = 0;
    const std::vector<uint8_t> shortcut =
        s.draw(glassOnLayerList(false, 0), true, passesShortcut, &copiesShortcut);
    const std::vector<uint8_t> pyramid =
        s.draw(glassOnLayerList(true, 0), true, passesPyramid, &copiesPyramid);
    CHECK(copiesShortcut == 0u);
    CHECK(copiesPyramid == 1u);
    std::printf("gl fewer passes, item 3: %zu passes with the pyramid, %zu without\n", passesPyramid,
                passesShortcut);
    CHECK(passesShortcut < passesPyramid);
    CHECK(shortcut == pyramid);
    size_t passesDrawn = 0;
    uint32_t copiesDrawn = 0;
    (void)s.draw(glassOnLayerList(true, 1), true, passesDrawn, &copiesDrawn);
    CHECK(copiesDrawn == 1u);  // over a real drawing the pyramid stays
}

TEST_CASE(
    "gl channel fold: shadows and glows are byte-identical; outer chains fold one pass (T-728, T-735)") {
    Session s;
    checkSame(s, "outer blur 6", {dropShadow(6, 0, false)}, 1);
    checkSame(s, "outer blur 24 (downsampled)", {dropShadow(24, 0, false)}, 1);
    checkSame(s, "outer blur 64 (eighth)", {dropShadow(64, 0, false)}, 1);
    checkSame(s, "inset blur 6", {dropShadow(6, 0, true)}, 0);
    checkSame(s, "inset blur 24 (downsampled)", {dropShadow(24, 0, true)}, 0);
    checkSame(s, "inset blur 64 (eighth)", {dropShadow(64, 0, true)}, 0);
    checkSame(s, "glow 8", {glow(8, false)}, 1);
    checkSame(s, "glow 30 (downsampled)", {glow(30, false)}, 1);
    checkSame(s, "inner glow 8", {glow(8, true)}, 0);
    checkSame(s, "inner glow 30 (downsampled)", {glow(30, true)}, 0);
    checkSame(s, "outer + inset + glow", {dropShadow(10, 0, false), dropShadow(12, 0, true), glow(6, false)},
              2);
    checkSame(s, "spread 2 outer, spread 3 inset", {dropShadow(6, 2, false), dropShadow(6, 3, true)}, 0);
}
