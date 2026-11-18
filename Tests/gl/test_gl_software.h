#pragma once

// T-733 / T-735: the whole-pixel backdrop-cache check of xt_gl_tests (a cached result against a fresh
// rasterisation at the new position) requires byte identity on a hardware GL implementation; on a
// CPU rasterizer (Mesa llvmpipe / softpipe, SwiftShader) a maximum channel difference of 1 is
// accepted. The renderer is recognized by its GL_RENDERER string.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gl_loader.h"

namespace xt::gl::test {

/// True when the current context's GL_RENDERER names a CPU rasterizer. Printed once.
inline bool softwareRenderer() {
    static const bool software = [] {
        const char* r = reinterpret_cast<const char*>(xt::gl::glGetString(xt::gl::GL_RENDERER));
        const char* name = r != nullptr ? r : "";
        const bool sw = std::strstr(name, "llvmpipe") != nullptr ||
                        std::strstr(name, "softpipe") != nullptr ||
                        std::strstr(name, "SwiftShader") != nullptr;
        std::printf("xt_gl_tests renderer: %s (%s)\n", name, sw ? "software" : "hardware");
        return sw;
    }();
    return software;
}

/// The largest per-channel difference between two RGBA8 buffers.
inline int maxChannelDiff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    int m = 0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        const int v = std::abs(int(a[i]) - int(b[i]));
        m = v > m ? v : m;
    }
    return m;
}

}  // namespace xt::gl::test
