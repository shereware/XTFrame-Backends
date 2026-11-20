// T-808 item 5: GlRenderer::shutdown() ("Releases every GL object created by init()") leaves no GL
// program behind. There is no GL resource counter, so the program names are enumerated with
// glIsProgram before init() and after shutdown(): every name that is a program afterwards was one
// before. Skips (does not fail) without a GL context, per test_gl_main.cpp's rule.

#include <SDL.h>

#include <cstdint>
#include <vector>

#include "doctest.h"
#include "gl_loader.h"
#include "xt/gl/gl_renderer.h"

using namespace xt;

namespace {

void* sdlLoadProcShutdown(const char* name) { return reinterpret_cast<void*>(SDL_GL_GetProcAddress(name)); }

/// The program names among 1..kScan that glIsProgram reports (glIsProgram is not in xt_gl's loader).
std::vector<uint32_t> liveProgramsT808() {
    using IsProgramFn = unsigned char (*)(uint32_t);  // declared like gl_loader.h's pointers
    const auto isProgram = reinterpret_cast<IsProgramFn>(SDL_GL_GetProcAddress("glIsProgram"));
    REQUIRE(isProgram != nullptr);
    constexpr uint32_t kScan = 4096;
    std::vector<uint32_t> out;
    for (uint32_t name = 1; name <= kScan; ++name) {
        if (isProgram(name) != 0) {
            out.push_back(name);
        }
    }
    return out;
}

}  // namespace

TEST_CASE("gl: T-808 GlRenderer::shutdown deletes every program init created") {
    const std::vector<uint32_t> before = liveProgramsT808();
    {
        gl::GlRenderer renderer;
        REQUIRE(renderer.init(sdlLoadProcShutdown));
        renderer.resize(64, 64);
        CHECK(liveProgramsT808().size() > before.size());  // the scan sees init's programs
        renderer.shutdown();
    }
    std::vector<uint32_t> leaked;
    for (const uint32_t name : liveProgramsT808()) {
        bool existed = false;
        for (const uint32_t b : before) {
            existed = existed || b == name;
        }
        if (!existed) {
            leaked.push_back(name);
        }
    }
    CAPTURE(leaked.size());
    CHECK(leaked.empty());
}
