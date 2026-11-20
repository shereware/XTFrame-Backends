#include <cstddef>

#include "gl_device.h"

/// T-806 (Docs/design/23-host-integration.md §23.aj): GlDevice's Shading::TexturePlanar draw, apart from
/// gl_device.cpp (kept within the file-length limit).

namespace xt::gl {

void GlDevice::drawPlanar(const gpu::DrawCmd& cmd) {
    // The host's R8 planes texture on unit 0 (its own wrap and filter are not relied on: the shader
    // clamps each plane), gpu_frame.h's PlanarParams as kTexturePlanarFrag's uPlanar.
    size_t avail = 0;
    const float* p = paramsAt(cmd.paramIndex, avail);
    gpu::PlanarParams pp{};
    if (!gpu::readPlanarParams(p, avail, pp)) {
        return;
    }
    Program& prog = m_texturePlanar;
    if (!useLeaf(prog, cmd)) return;
    prog.shadow.set4fv(prog.uPlanar, static_cast<int>(sizeof(pp) / (4 * sizeof(float))), &pp.uv[0][0]);
    bindSampled(cmd.texture, cmd.sampler);
    m_geometry.drawRange(cmd.firstIndex, cmd.indexCount, cmd.baseVertex, m_cache.stats);
}

}  // namespace xt::gl
