// T-730 (Docs/design/10-backends.md §10.18 "B 定案" second step, Docs/tasks/T-730.md "測試"): the cases
// of Tests/wgpu/test_wgpu_backdrop_cache.cpp on xt_gl. One frame builder keeps its effect cache across the
// frames (the pair of a glass panel is stored on the second build and pasted from the third); every frame
// is compared with the same list built by a fresh builder (no cache) and drawn by the same device: byte
// for byte at rest, under a backdrop that moves and a whole number of pixels away.

#include <SDL.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "test_gl_software.h"
#include "xt/command_list.h"
#include "xt/gl/gl_renderer.h"
#include "xt/gpu/frame_builder.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/types.h"

using namespace xt;

namespace {

constexpr int kW = 256, kH = 192;

void* sdlLoadProcCache(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

/// A GlRenderer and its own single-sample RGBA8 + stencil FBO; `builder` keeps its cache across draws.
struct CacheSession {
    gl::GLuint fbo = 0, color = 0, rbo = 0;
    gl::GlRenderer renderer;
    gpu::FrameBuilder builder;
    gpu::FrameBuilder::BuildStats stats;

    CacheSession() {
        REQUIRE(renderer.init(sdlLoadProcCache));
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
        REQUIRE(renderer.device() != nullptr);
        builder.init(*renderer.device());
        builder.resize(kW, kH);
        builder.setEdgeFringe(true);
    }
    ~CacheSession() {
        builder.shutdown();
        renderer.shutdown();
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteRenderbuffers(1, &rbo);
        glDeleteTextures(1, &color);
    }
    CacheSession(const CacheSession&) = delete;
    CacheSession& operator=(const CacheSession&) = delete;

    std::vector<uint8_t> drawWith(gpu::FrameBuilder& b, const CommandList& cl) {
        using namespace xt::gl;
        gpu::GpuFrame frame;
        b.build(cl, nullptr, nullptr, nullptr, frame);
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
        return px;
    }

    /// `cl` through the session's builder (its cache) ...
    std::vector<uint8_t> draw(const CommandList& cl) {
        std::vector<uint8_t> px = drawWith(builder, cl);
        stats = builder.lastBuildStats();
        return px;
    }
    /// ... and through a fresh one (the segment in full).
    std::vector<uint8_t> uncached(const CommandList& cl) {
        gpu::FrameBuilder fresh;
        fresh.init(*renderer.device());
        fresh.resize(kW, kH);
        fresh.setEdgeFringe(true);
        std::vector<uint8_t> px = drawWith(fresh, cl);
        CHECK(fresh.lastBuildStats().backdropPairHits == 0u);
        fresh.shutdown();
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

void fill(CommandList& cl, PathRef path, Color color, Mat2x3 xf = Mat2x3{}) {
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

EffectInstance fx(EffectKind kind, std::initializer_list<float> params, Color color) {
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

/// Stripes offset by `stripes`, then a glass panel whose effect bounds start at `at`.
CommandList glassList(Vec2 at, float stripes) {
    CommandList cl;
    cl.viewport = Rect{0, 0, kW, kH};
    cl.devicePixelRatio = 1;
    for (int i = 0; i < 10; ++i) {
        const float x = stripes + static_cast<float>(i) * 32.0f - 32.0f;
        fill(cl, polygon(cl, {Vec2{x, 0}, Vec2{x + 32, 0}, Vec2{x + 32, kH}, Vec2{x, kH}}),
             Color{0.1f * static_cast<float>(i), 0.35f, 0.9f - 0.08f * static_cast<float>(i), 1});
    }
    const float margin = 30, w = 128, h = 80;
    const Mat2x3 world = translation(Vec2{at.x + margin, at.y + margin});
    EffectCmd e{};
    e.effectOffset = static_cast<uint32_t>(cl.effects.size());
    cl.effects.push_back(fx(EffectKind::BackdropBlur, {10, 0.4f, 0.1f}, Color{0, 0, 0, 1}));
    cl.effects.push_back(fx(EffectKind::DropShadow, {3, 7, 12, 0, 1, 0}, Color{0.05f, 0, 0.1f, 0.75f}));
    cl.effects.push_back(fx(EffectKind::DropShadow, {0, 3, 8, 0, 1, 1}, Color{0, 0, 0.2f, 0.6f}));
    e.effectCount = 3;
    e.xf = world;
    e.boundsHint = Rect{at.x, at.y, w + 2 * margin, h + 2 * margin};
    e.contentBounds = Rect{at.x + margin, at.y + margin, w, h};
    cl.effectCmds.push_back(e);
    cl.commands.push_back(Command{CmdKind::PushEffect, {}, static_cast<uint32_t>(cl.effectCmds.size() - 1)});
    fill(cl, polygon(cl, {Vec2{0, 0}, Vec2{w, 0}, Vec2{w, h}, Vec2{0, h}}), Color{1, 1, 1, 0.5f}, world);
    fill(cl, polygon(cl, {Vec2{26.3f, 12.6f}, Vec2{106.2f, 22.1f}, Vec2{56.7f, 70.4f}}),
         Color{0.2f, 0.9f, 0.6f, 0.65f}, world);
    cl.commands.push_back(Command{CmdKind::PopEffect, {}, 0});
    return cl;
}

size_t differing(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t n = 0;
    for (size_t i = 0; i + 3 < a.size() && i + 3 < b.size(); i += 4) {
        n += (a[i] != b[i] || a[i + 1] != b[i + 1] || a[i + 2] != b[i + 2] || a[i + 3] != b[i + 3]) ? 1u : 0u;
    }
    return n;
}

}  // namespace

TEST_CASE("gl backdrop cache: T-730 a glass panel at rest and under a moving backdrop, byte for byte") {
    CacheSession s;
    for (int i = 0; i < 6; ++i) {
        CAPTURE(i);
        const CommandList cl = glassList(Vec2{30, 20}, i >= 3 ? static_cast<float>(i) * 5.0f : 0.0f);
        const std::vector<uint8_t> px = s.draw(cl);
        if (i >= 2) {
            CHECK(s.stats.backdropPairHits == 1u);
        }
        CHECK(differing(px, s.uncached(cl)) == 0u);
    }
}

TEST_CASE("gl backdrop cache: T-730 a glass panel moving by whole pixels, byte for byte") {
    CacheSession s;
    for (int i = 0; i < 5; ++i) {
        CAPTURE(i);
        const CommandList cl = glassList(Vec2{30.0f + float(i), 20.0f + float(i / 2)}, 0.0f);
        const std::vector<uint8_t> px = s.draw(cl);
        if (i >= 2) {
            CHECK(s.stats.backdropPairHits == 1u);
        }
        // T-733 / T-735: the cached result was rasterised at the old position and pasted exactly (nearest
        // texel); the reference is rasterised afresh at the new one. Antialiased edges are translation
        // invariant bit for bit on hardware, not on a CPU rasterizer: there 1 in a channel is accepted.
        const std::vector<uint8_t> ref = s.uncached(cl);
        const size_t differ = differing(px, ref);
        const int maxChannel = xt::gl::test::maxChannelDiff(px, ref);
        std::printf("gl backdrop cache whole-pixel move %d: %zu px differ, max channel %d\n", i, differ,
                    maxChannel);
        if (xt::gl::test::softwareRenderer()) {
            CHECK(maxChannel <= 1);
        } else {
            CHECK(differ == 0u);
        }
    }
}
