// T-806 (Docs/design/23-host-integration.md §23.aj, Docs/tasks/T-806.md "測試"): planar external
// textures through xt_gl -- the cases of Tests/wgpu/test_wgpu_planar.cpp (Tests/common/planar_scenes.h)
// with host textures made the way a host makes them: plain glTexImage2D. The planes' GL_R8 texture keeps
// its default parameters (wrap GL_REPEAT, a mipmap min filter), so nothing but the shader's per-plane
// clamp keeps a sample inside its plane; the GL_RGBA8 reference is clamped to its edges. Skips (does not
// fail) without a GL context, per test_gl_main.cpp's rule.

#include <SDL.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "planar_scenes.h"
#include "test_gl_software.h"
#include "xt/gl/gl_renderer.h"

using namespace xt;
namespace pt = xt::planar_test;

namespace {

void* sdlLoadProcPlanar(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

class GlBackend : public pt::Backend {
public:
    GlBackend() {
        REQUIRE(m_renderer.init(sdlLoadProcPlanar));
        using namespace xt::gl;
        glGenTextures(1, &m_color);
        glBindTexture(GL_TEXTURE_2D, m_color);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8), pt::kFrameW, pt::kFrameH, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, static_cast<GLint>(GL_NEAREST));
        glGenRenderbuffers(1, &m_rbo);
        glBindRenderbuffer(GL_RENDERBUFFER, m_rbo);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, pt::kFrameW, pt::kFrameH);
        glGenFramebuffers(1, &m_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_color, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_rbo);
        REQUIRE(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE);
        m_renderer.resize(pt::kFrameW, pt::kFrameH);
    }
    ~GlBackend() override {
        m_renderer.shutdown();
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &m_fbo);
        glDeleteRenderbuffers(1, &m_rbo);
        glDeleteTextures(1, &m_color);
        for (GLuint t : m_textures) {
            glDeleteTextures(1, &t);
        }
    }
    GlBackend(const GlBackend&) = delete;
    GlBackend& operator=(const GlBackend&) = delete;

    const char* name() const override { return "gl"; }
    uint32_t texture(int w, int h, bool r8, const uint8_t* data) override {
        using namespace xt::gl;
        GLuint t = 0;
        glGenTextures(1, &t);
        glBindTexture(GL_TEXTURE_2D, t);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(r8 ? GL_R8 : GL_RGBA8), w, h, 0,
                     r8 ? GL_RED : GL_RGBA, GL_UNSIGNED_BYTE, data);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (!r8) {
            // The RGBA8 reference is the picture drawn right: clamped at its edges (GL_REPEAT would blend
            // the opposite edge into a scaled image's border). The planes' texture keeps GL_REPEAT.
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, static_cast<GLint>(GL_CLAMP_TO_EDGE));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, static_cast<GLint>(GL_CLAMP_TO_EDGE));
        }
        m_textures.push_back(t);
        return t;
    }
    std::vector<uint8_t> render(const CommandList& list, AssetResolver& assets, const IrView& ir) override {
        using namespace xt::gl;
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glViewport(0, 0, pt::kFrameW, pt::kFrameH);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilMask(0xFFu);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClearStencil(0);
        glClear(GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        m_renderer.render(list, nullptr, &assets, &ir);
        m_error = m_renderer.lastError();
        const size_t row = size_t(pt::kFrameW) * 4;
        std::vector<uint8_t> bottomUp(row * pt::kFrameH), out(bottomUp.size());
        glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, pt::kFrameW, pt::kFrameH, GL_RGBA, GL_UNSIGNED_BYTE, bottomUp.data());
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        for (int y = 0; y < pt::kFrameH; ++y) {
            std::copy_n(bottomUp.begin() + static_cast<long>(size_t(pt::kFrameH - 1 - y) * row), row,
                        out.begin() + static_cast<long>(size_t(y) * row));
        }
        return out;
    }
    std::string error() override { return m_error; }
    bool softwareAdapter() override { return gl::test::softwareRenderer(); }  // T-812

private:
    gl::GlRenderer m_renderer;
    gl::GLuint m_fbo = 0, m_color = 0, m_rbo = 0;
    std::vector<gl::GLuint> m_textures;
    std::string m_error;
};

}  // namespace

TEST_CASE("gl planar: every leaf image path matches the RGBA8 picture within the 8-bit 4:2:0 tolerance") {
    GlBackend backend;
    pt::runPixelCases(backend);
}

TEST_CASE("gl planar: no plane bleeds into its neighbour, with the host texture's wrap left at GL_REPEAT") {
    GlBackend backend;
    pt::runBleedCase(backend);
}
