#include <cstdint>

#include "wgpu_internal.h"
#include "xt/wgpu/wgpu_device.h"

/// WgpuDevice's world-layer targets (xt/wgpu/wgpu_device.h; Docs/design/33-world-layers.md §33.8,
/// T-701): split from wgpu_device.cpp.

namespace xt::wgpu {

using detail::makeTexture;
using detail::makeView;
using detail::releaseTexture;

gpu::TextureHandle WgpuDevice::acquireLayerTarget(uint32_t layerId, uint16_t w, uint16_t h) {
    return acquireLayerTargetWithStencil(layerId, w, h, true);
}

gpu::TextureHandle WgpuDevice::acquireLayerTargetWithStencil(uint32_t layerId, uint16_t w, uint16_t h,
                                                             bool stencil) {
    (void)layerId;
    if (m_device == nullptr || w == 0 || h == 0 ||
        (m_maxTextureSize != 0 && (w > m_maxTextureSize || h > m_maxTextureSize))) {
        return 0;
    }
    Texture t;
    t.texture = makeTexture(m_device, "xt_wgpu world layer", w, h, WGPUTextureFormat_RGBA8Unorm, 1,
                            WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_TextureBinding |
                                WGPUTextureUsage_CopySrc | WGPUTextureUsage_CopyDst);
    if (t.texture != nullptr) countTextureCreate();
    t.view = makeView(t.texture);
    if (t.view == nullptr) {
        releaseTexture(t.texture, t.view);
        return 0;
    }
    t.width = w;
    t.height = h;
    t.layerNoStencil = !stencil;
    return addTexture(t);
}

void WgpuDevice::releaseLayerTarget(gpu::TextureHandle target) { destroyTexture(target); }

void WgpuDevice::releaseLayerAttachments() {
    for (LayerAttachments& a : m_layerAttachments) {
        releaseTexture(a.msaa, a.msaaView);
        releaseTexture(a.stencil, a.stencilView);
    }
    m_layerAttachments.clear();
}

void WgpuDevice::submitLayer(gpu::TextureHandle target, const gpu::GpuFrame& frame) {
    const Texture* t = find(target);
    if (m_device == nullptr || t == nullptr || t->texture == nullptr || t->external || t->cached ||
        t->format != gpu::TextureFormat::Rgba8) {
        m_lastError = "submitLayer: not a layer target of this device";
        return;
    }
    // The surface's own state, put back below.
    const WGPUTextureView surface = m_surface;
    const int width = m_width, height = m_height;
    const bool clearSurface = m_clearSurface;
    const WGPUColor clearColor = m_clearColor;
    const WGPUTextureFormat targetFormat = m_targetFormat;
    const WGPUTexture surfaceMsaa = m_surfaceMsaa;
    const WGPUTextureView surfaceMsaaView = m_surfaceMsaaView;
    const WGPUTexture surfaceStencil = m_surfaceStencil;
    const WGPUTextureView surfaceStencilView = m_surfaceStencilView;
    const int attachW = m_surfaceAttachW, attachH = m_surfaceAttachH;
    const bool stencil = !t->layerNoStencil;

    m_surface = t->view;
    m_width = t->width;
    m_height = t->height;
    m_clearSurface = true;
    m_clearColor = WGPUColor{0, 0, 0, 0};
    m_targetFormat = WGPUTextureFormat_RGBA8Unorm;
    const bool own = m_sharing == nullptr;
    if (own) {
        // This device's attachments are the surface's size: the layer gets a set of its own.
        m_surfaceMsaa = m_surfaceStencil = nullptr;
        m_surfaceMsaaView = m_surfaceStencilView = nullptr;
        m_surfaceAttachW = m_surfaceAttachH = 0;
        for (size_t i = 0; i < m_layerAttachments.size(); ++i) {
            LayerAttachments& a = m_layerAttachments[i];
            if (a.width != m_width || a.height != m_height || (a.stencil != nullptr) != stencil) continue;
            m_surfaceMsaa = a.msaa;
            m_surfaceMsaaView = a.msaaView;
            m_surfaceStencil = a.stencil;
            m_surfaceStencilView = a.stencilView;
            m_surfaceAttachW = a.width;
            m_surfaceAttachH = a.height;
            m_layerAttachments.erase(m_layerAttachments.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
    }

    m_surfaceNoStencil = !stencil;  // T-703
    submit(frame);
    m_surfaceNoStencil = false;

    ++m_layerSubmits;
    if (own) {
        if (m_surfaceStencil != nullptr || m_surfaceMsaa != nullptr) {
            // Kept for the next frame of a layer this size; the least recently used set goes.
            if (m_layerAttachments.size() >= kLayerAttachmentSets) {
                size_t oldest = 0;
                for (size_t i = 1; i < m_layerAttachments.size(); ++i) {
                    if (m_layerAttachments[i].lastUsed < m_layerAttachments[oldest].lastUsed) oldest = i;
                }
                releaseTexture(m_layerAttachments[oldest].msaa, m_layerAttachments[oldest].msaaView);
                releaseTexture(m_layerAttachments[oldest].stencil, m_layerAttachments[oldest].stencilView);
                m_layerAttachments.erase(m_layerAttachments.begin() + static_cast<std::ptrdiff_t>(oldest));
            }
            m_layerAttachments.push_back(LayerAttachments{m_surfaceAttachW, m_surfaceAttachH, m_surfaceMsaa,
                                                          m_surfaceMsaaView, m_surfaceStencil,
                                                          m_surfaceStencilView, m_layerSubmits});
        }
        m_surfaceMsaa = surfaceMsaa;
        m_surfaceMsaaView = surfaceMsaaView;
        m_surfaceStencil = surfaceStencil;
        m_surfaceStencilView = surfaceStencilView;
        m_surfaceAttachW = attachW;
        m_surfaceAttachH = attachH;
    }
    m_surface = surface;
    m_width = width;
    m_height = height;
    m_clearSurface = clearSurface;
    m_clearColor = clearColor;
    m_targetFormat = targetFormat;
}

}  // namespace xt::wgpu
