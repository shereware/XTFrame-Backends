#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "capi_internal.h"

#ifdef XT_CAPI_HAS_WGPU
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

#include <cmath>
#include <span>

#include "xt/dds.h"
#include "xt/gpu/gpu_frame.h"
#include "xt/gpu/image_texture.h"
#include "xt/gpu/world_layers.h"
#include "xt/ir.h"
#include "xt/wgpu/wgpu_device.h"
#include "xt/wgpu/wgpu_inflight.h"
#include "xt/wgpu/wgpu_renderer.h"
#include "xt/wgpu/wgpu_shared.h"
#include "xt/wgpu/wgpu_window.h"
#if defined(XT_CAPI_HAS_TEXT) && !defined(XT_GPU_NO_TEXT)
#define XT_CAPI_GPU_ATLAS 1
#include "xt/text/glyph_atlas.h"
#endif
#endif

/// `xt_gpu_target_*`, `xt_surface_render_gpu`, `xt_surface_detach_gpu` (Docs/design/50-c-abi.md §50.z,
/// Docs/tasks/T-636.md). A target owns a WebGPU device (WgpuInit::headless, 10 §10.10.1) and either a
/// window surface (WgpuWindowSurface) or an offscreen RGBA8 texture. Each attached surface renders its
/// CommandList with its own WgpuRenderer into its own texture of the surface's size; the target then
/// composites that texture into `dest` with a one-command GpuFrame (Shading::Texture, LinearClamp,
/// SrcOver) submitted to its composer's WgpuDevice -- the existing draw path, no shader of its own.
/// T-648 (50 §50.z addendum): the target owns a WgpuSharedCache that its composer and every attached
/// surface's renderer borrow -- one pipeline set (compiled once per target) and one texture per bundle
/// image content. T-656 (50 §50.ab item 1): a texture no surface uses any more (detach, trim, destroy)
/// goes to the cache's idle list, bounded by the idle budget (xt_gpu_target_set_image_budget, default
/// 64 MiB); xt_gpu_target_trim empties it and xt_gpu_target_memory_stats reports it. Each surface keeps
/// its own glyph atlas (starting at 1024 x 1024, growing to 2048 x 2048). T-675 (50 §50.ac item 2): the
/// transient pool, the surface attachment sets and the dummy are the target's (in its WgpuSharedCache),
/// and a surface takes its texture from the target's pool of surface textures and gives it back when
/// hidden; a hidden surface keeps its renderer (xt_surface_trim, detach and destroy release it). T-683
/// (50 §50.ad items 2-3): xt_gpu_target_create prewarms every built-in shader for the surfaces' format and
/// sample count, and a hidden surface also releases its coverage atlas (its glyph atlas stays).
/// Without wgpu (XT_BUILD_WGPU=OFF, the XTFrame-Runtime mirror) capi_gpu_stub.cpp replaces this file and
/// every function only reports "built without wgpu" (45 §45.8 item 2, T-689).

using xt::capi::fail;

#ifdef XT_CAPI_HAS_WGPU

/// The object behind `xt_gpu_target*`.
struct xt_gpu_target {
    xt::wgpu::WgpuInit init;             ///< instance / adapter / device / queue (owned)
    xt::wgpu::WgpuWindowSurface window;  ///< valid for a window target
    WGPUTexture offscreen = nullptr;     ///< XT_WINDOW_NONE target
    WGPUTextureFormat format = WGPUTextureFormat_RGBA8Unorm;
    /// T-648: pipelines and image textures shared by the composer and the surfaces' renderers;
    /// declared before them, and shut down by the destructor once they are.
    xt::wgpu::WgpuSharedCache shared;
    /// T-648: the shared-cache counters as last logged (xt_surface_render_gpu logs on a change).
    xt::wgpu::WgpuSharedStats logged;
    /// T-656: the GPU resources of the attached surfaces that have them (xt_gpu_target_memory_stats);
    /// each SurfaceGpu adds itself when built and removes itself when destroyed.
    std::vector<xt::capi::SurfaceGpu*> gpus;
    xt::wgpu::WgpuRenderer composer;  ///< its WgpuDevice composites; also readback
    int width = 0, height = 0;
    uint32_t sampleCount = 1;  ///< T-647: the attached surfaces' renderers (1 or 4; T-666: 0 = 1)
    bool inFrame = false;
    /// T-658 (sheet item 3): begin_frame's clear, submitted with the frame's first composite (or by
    /// end_frame / readback when nothing was composited before them) -- one submit instead of two.
    bool clearPending = false;
    WGPUColor pendingClear = {0, 0, 0, 0};
    /// T-658: the submits and render passes made on this target so far (composer and surface renderers);
    /// logged when a surface stops or starts drawing (tests read them there).
    uint64_t submits = 0, passes = 0;
    /// T-674 (50 §50.ac item 1): at most two frames in flight -- every submit on the target's device
    /// (composer and surface renderers) notes its index; begin_frame waits for the frame two back.
    xt::wgpu::WgpuFramesInFlight inflight;
    WGPUTexture frameTexture = nullptr;   ///< the frame's texture (window's or `offscreen`), not owned
    WGPUTextureView frameView = nullptr;  ///< the frame's view (owned for `offscreen` only)
    uint32_t attached = 0;                ///< surfaces attached (destroy refuses while > 0)
    xt::gpu::GpuFrame frame;              ///< composition scratch
    /// T-709 (50 §50.af): which frame this is -- a number unique across all targets of the process, taken
    /// by begin_frame (0 before the first), so a surface can tell "this frame's render_layers_gpu".
    uint64_t frameSerial = 0;

    ~xt_gpu_target() {
        releaseFrame();
        composer.shutdown();
        shared.shutdown();  // every surface is detached (destroy refuses otherwise)
        window.release();
        releaseOffscreen();
        init.release();
    }
    bool windowed() const { return window.valid(); }
    void releaseFrame() {
        if (windowed()) {
            window.releaseFrame();
        } else if (frameView != nullptr) {
            wgpuTextureViewRelease(frameView);
        }
        frameView = nullptr;
        frameTexture = nullptr;
    }
    void releaseOffscreen() {
        if (offscreen != nullptr) {
            wgpuTextureDestroy(offscreen);
            wgpuTextureRelease(offscreen);
            offscreen = nullptr;
        }
    }
    bool createOffscreen() {
        releaseOffscreen();
        offscreen = xt::wgpu::createRenderTexture(init.device, width, height, format, "xt_capi gpu target");
        return offscreen != nullptr;
    }
};

namespace xt::capi {

/// T-709 (50 §50.af item 2): the world layers of one attached surface -- a gpu::WorldLayerBuilder on the
/// surface renderer's device (its layer targets are that device's textures), a glyph atlas of its own
/// (the builder's text needs an atlas no other FrameBuilder uses, world_layers.h), and the composer's
/// external-texture entries naming the layer targets that draw_layers_gpu draws with. Made by the first
/// render_layers_gpu; released by trim, trim-on-hide, detach, destroy and set_world_layers(s, 0).
struct SurfaceLayers {
    gpu::WorldLayerBuilder builder;
#ifdef XT_CAPI_GPU_ATLAS
    std::unique_ptr<text::GlyphAtlas> atlas;
#endif
    /// The last build gave a layer list (count = builder.layers().size()); false = no layers.
    bool valid = false;
    /// One layer target as the composer samples it.
    struct Bound {
        gpu::TextureHandle layer = 0;  ///< the surface device's handle
        WGPUTexture texture = nullptr;
        WGPUTextureView view = nullptr;  ///< owned
        gpu::TextureHandle composer = 0;
        uint16_t width = 0, height = 0;
    };
    std::vector<Bound> bound;
    std::vector<gpu::WorldLayer> quadLayers;  ///< draw_layers_gpu scratch: the layers with composer handles
};

/// The GPU resources of one attached surface: its renderer on the target's device and the texture it
/// renders into, registered with the target's composer as an external texture.
struct SurfaceGpu {
    xt_gpu_target* target = nullptr;
    wgpu::WgpuRenderer renderer;
#ifdef XT_CAPI_GPU_ATLAS
    std::unique_ptr<text::GlyphAtlas> atlas;
#endif
    WGPUTexture texture = nullptr;
    WGPUTextureView view = nullptr;
    gpu::TextureHandle handle = 0;
    int width = 0, height = 0;
    uint32_t executed = 0, skipped = 0;
    /// T-709: the world layers (null until the first render_layers_gpu; never made for a flat surface).
    std::unique_ptr<SurfaceLayers> layers;
    /// T-658: the size the renderer was last resized to (its builder and device).
    int renderWidth = 0, renderHeight = 0;
    /// T-658 (50 §50.ab item 3 ii / iii): a frame was built since these resources were made, and whether that
    /// frame had no draw command -- such a surface is neither drawn nor composited, and its texture (and
    /// the renderer's 4x attachments) are not made until a frame has something to draw.
    bool built = false;
    bool empty = false;
    /// T-662 (10 §10.11 item 2, supplement 2): the last build held a coverage shape back for the
    /// frame budget (`WgpuRenderer::wantsAnotherFrame`), so the next render builds again even
    /// without damage, as if the whole surface were dirty.
    bool repaint = false;

    ~SurfaceGpu() {
        if (target != nullptr) {  // T-656
            target->gpus.erase(std::remove(target->gpus.begin(), target->gpus.end(), this),
                               target->gpus.end());
        }
        releaseLayers();
        releaseTexture();
        renderer.shutdown();
    }
    /// T-709: the composer's entries for the layer targets (the views go too).
    void unbindLayers() {
        if (layers == nullptr) return;
        for (const SurfaceLayers::Bound& b : layers->bound) {
            if (b.composer != 0 && target != nullptr && target->composer.device() != nullptr) {
                target->composer.device()->unregisterExternalTexture(b.composer);
            }
            if (b.view != nullptr) wgpuTextureViewRelease(b.view);
        }
        layers->bound.clear();
    }
    /// T-709 (50 §50.af item 2): every layer target and texture, the layers' atlas and the composer's
    /// entries; the next render_layers_gpu makes them again.
    void releaseLayers() {
        if (layers == nullptr) return;
        unbindLayers();
        layers->builder.shutdown();  // the targets go back to the renderer's device, still alive here
        layers.reset();
    }
    /// T-675 (50 §50.ac item 2): the texture goes back to the target's pool of surface textures.
    void releaseTexture() {
        if (handle != 0 && target != nullptr && target->composer.device() != nullptr) {
            target->composer.device()->unregisterExternalTexture(handle);
        }
        handle = 0;
        if (texture != nullptr && target != nullptr) {
            target->shared.releaseSurfaceTexture(texture, view);
        } else if (view != nullptr) {
            wgpuTextureViewRelease(view);
        }
        view = nullptr;
        texture = nullptr;
        width = height = 0;
    }
    /// The texture at `w` x `h` (another one when the size changed; T-675: from the target's pool, made
    /// only when it has none of that size). False when it cannot be had.
    bool ensureTexture(int w, int h, bool& fresh) {
        fresh = false;
        if (texture != nullptr && handle != 0 && w == width && h == height) return true;
        releaseTexture();
        texture = target->shared.acquireSurfaceTexture(w, h, WGPUTextureFormat_RGBA8Unorm, view);
        if (texture == nullptr) return false;
        handle = target->composer.device()->registerExternalTexture(view, static_cast<uint16_t>(w),
                                                                    static_cast<uint16_t>(h));
        width = w;
        height = h;
        fresh = true;
        return handle != 0;
    }
    /// Resizes the renderer to `w` x `h` when it is not that size yet. True when it was resized.
    bool ensureRenderSize(int w, int h) {
        if (w == renderWidth && h == renderHeight) return false;
        renderer.resize(w, h);
        renderWidth = w;
        renderHeight = h;
        return true;
    }
};

void SurfaceGpuDeleter::operator()(SurfaceGpu* gpu) const noexcept { delete gpu; }

void detachGpu(xt_surface& surface) {
    surface.gpu.reset();  // needs the target (alive: destroy refuses while attached)
    if (surface.gpuTarget != nullptr && surface.gpuTarget->attached > 0) {
        --surface.gpuTarget->attached;
    }
    surface.gpuTarget = nullptr;
    surface.renderPath = RenderPath::None;
    // A later xt_surface_render repaints its whole buffer (the damage since its last SW frame is lost).
    surface.target = sw::SwTarget{};
}

void releaseGpuResources(xt_surface& surface, bool everything) {
    if (surface.gpu == nullptr) return;
    if (!everything) {
        // T-675 (50 §50.ac item 2): trim-on-hide gives back the surface texture and the effect-cache
        // targets (to the target's pools) and ends its image uses; the renderer and its small resources
        // stay, so showing it again makes none of them. T-683 (§50.ad item 3): its coverage atlas is
        // released too (the glyph atlas stays).
        SurfaceGpu& g = *surface.gpu;
        releaseGpuPoolUse(surface);  // T-685
        g.releaseLayers();           // T-709 (50 §50.af item 2): hidden, the layer targets go
        g.releaseTexture();
        g.renderer.trim();
        g.built = false;  // the next frame is built in full (the Context reports `full` too)
        log(0, "xt_surface: GPU textures returned to the target (" + surface.uiId + ")");
        return;
    }
    surface.gpu.reset();  // T-641: gpuTarget, `attached` and renderPath stay (still attached)
    // T-675: its textures go back to the target's pools; the rest goes back to the target's device,
    // whose allocator keeps the memory for the next resources.
    log(0, "xt_surface: GPU resources released (" + surface.uiId + ")");
}

void releaseGpuPoolUse(xt_surface& surface) {
    if (surface.gpu == nullptr) return;
    if (wgpu::WgpuDevice* d = surface.gpu->renderer.device(); d != nullptr) d->releasePoolUse();
}

bool gpuDrawCounts(const xt_surface& surface, uint32_t& executed, uint32_t& skipped) {
    if (surface.gpu == nullptr) return false;
    executed = surface.gpu->executed;
    skipped = surface.gpu->skipped;
    return true;
}

bool gpuWantsAnotherFrame(const xt_surface& surface) {
    return surface.gpu != nullptr && surface.gpu->repaint;
}

}  // namespace xt::capi

namespace {

/// Straight 0..1 clear colour -> premultiplied; NULL = transparent.
WGPUColor clearColor(const float* rgba) {
    if (rgba == nullptr) return WGPUColor{0, 0, 0, 0};
    auto unit = [](float v) { return std::isfinite(v) ? std::clamp(static_cast<double>(v), 0.0, 1.0) : 0.0; };
    const double a = unit(rgba[3]);
    return WGPUColor{unit(rgba[0]) * a, unit(rgba[1]) * a, unit(rgba[2]) * a, a};
}

/// Submits `t->frame` into the frame's view with the composer's device, clearing first when
/// begin_frame's clear is still pending (T-658: the clear rides on the frame's first composite; the
/// same pixels as a clear pass of its own, then a loaded pass). Empty string, or the device's error.
std::string submitComposite(xt_gpu_target& t) {
    xt::wgpu::WgpuDevice* dev = t.composer.device();
    if (dev == nullptr || t.frameView == nullptr) return std::string();
    dev->setSurface(t.frameView);
    dev->setSurfaceClear(t.clearPending, t.pendingClear);
    t.clearPending = false;
    t.frame.targetWidth = t.width;
    t.frame.targetHeight = t.height;
    dev->submit(t.frame);
    dev->setSurfaceClear(false, WGPUColor{0, 0, 0, 0});
    dev->setSurface(nullptr);
    if (dev->lastSubmissionIndex() != 0) t.inflight.noteSubmit(dev->lastSubmissionIndex());  // T-674
    ++t.submits;
    t.passes += dev->lastStats().passes;
    return dev->lastError();
}

/// T-658: begin_frame's clear on its own when no composite carried it (end_frame, readback).
std::string flushPendingClear(xt_gpu_target& t) {
    if (!t.clearPending) return std::string();
    t.frame.clear();  // no pass: the device's own clear pass
    return submitComposite(t);
}

/// One textured quad of `handle` over `dest` (target pixels), scissored to the target.
bool buildCompositeFrame(xt_gpu_target& t, xt::gpu::TextureHandle handle, const xt_rect& dest) {
    xt::gpu::GpuFrame& f = t.frame;
    f.clear();
    const int cx0 = std::clamp(dest.x, 0, t.width), cy0 = std::clamp(dest.y, 0, t.height);
    const int cx1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(dest.x) + dest.w, 0, t.width));
    const int cy1 = static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(dest.y) + dest.h, 0, t.height));
    if (cx1 <= cx0 || cy1 <= cy0) return false;  // entirely outside the target
    const float x0 = static_cast<float>(dest.x), y0 = static_cast<float>(dest.y);
    const float x1 = x0 + static_cast<float>(dest.w), y1 = y0 + static_cast<float>(dest.h);
    const uint32_t white = 0xFFFFFFFFu;
    f.vertices = {{x0, y0, 1, 0, 0, white},
                  {x1, y0, 1, 1, 0, white},
                  {x1, y1, 1, 1, 1, white},
                  {x0, y1, 1, 0, 1, white}};
    f.indices = {0, 1, 2, 0, 2, 3};
    xt::gpu::DrawCmd cmd;
    cmd.indexCount = 6;
    cmd.vertexCount = 4;
    cmd.texture = handle;
    cmd.shading = xt::gpu::Shading::Texture;
    cmd.blend = xt::gpu::Blend::SrcOver;
    cmd.sampler = xt::gpu::Sampler::LinearClamp;
    cmd.clipX = static_cast<int16_t>(std::min(cx0, 32767));
    cmd.clipY = static_cast<int16_t>(std::min(cy0, 32767));
    cmd.clipW = static_cast<int16_t>(std::min(cx1 - cx0, 32767));
    cmd.clipH = static_cast<int16_t>(std::min(cy1 - cy0, 32767));
    f.cmds.push_back(cmd);
    xt::gpu::Pass pass;
    pass.target = xt::gpu::kSurfaceTarget;
    pass.cmdCount = 1;
    f.passes.push_back(pass);
    return true;
}

/// The largest surface / target side a texture handle and a scissor can describe.
constexpr int kMaxSide = 16384;

#ifdef XT_CAPI_GPU_ATLAS
/// T-648 (50 §50.z addendum): a surface's glyph atlas pages -- the first one starts at kAtlasFirstPage
/// and grows to kAtlasPage when full.
constexpr int kAtlasPage = 2048;
constexpr int kAtlasFirstPage = 1024;
#endif

/// T-648: logs the target's shared-cache counters when a texture was created, reused or freed, or a
/// pipeline set made or borrowed, since the last line (info level; tests count them).
void logSharedCache(xt_gpu_target& t) {
    const xt::wgpu::WgpuSharedStats s = t.shared.stats();
    const xt::wgpu::WgpuSharedStats& l = t.logged;
    if (s.imageCreates == l.imageCreates && s.imageReuses == l.imageReuses && s.imageFrees == l.imageFrees &&
        s.pipelineSets == l.pipelineSets && s.pipelineBorrows == l.pipelineBorrows &&
        s.textureCreates == l.textureCreates && s.poolFrees == l.poolFrees &&
        s.attachmentSets == l.attachmentSets && s.surfaceTexturesIdle == l.surfaceTexturesIdle &&
        s.shaderCompiles == l.shaderCompiles && s.pipelineCreates == l.pipelineCreates) {
        return;
    }
    t.logged = s;
    // T-675: then every texture created on the target's device so far and the target-level pools
    // (50 §50.ac item 2; tests and xt_gpu_mem_probe count the creates per scene switch here). T-683
    // (§50.ad item 2): the pipeline set's shader compiles and pipeline creates since the prewarm. T-685
    // (§50.ac addendum 2): last, the transient bytes the budget may not evict, T-696 (§50.ac addendum 3
    // item 6) those whose user still holds them but whose protection expired, and the budget / ageing
    // evictions so far.
    xt::capi::log(
        0, "xt_gpu_target: shared image textures " + std::to_string(s.imageTextures) + " (" +
               std::to_string(s.imageBytes / 1024u) + " KiB; uploaded " + std::to_string(s.imageCreates) +
               ", reused " + std::to_string(s.imageReuses) + ", freed " + std::to_string(s.imageFrees) +
               "); pipeline sets " + std::to_string(s.pipelineSets) + ", borrowed " +
               std::to_string(s.pipelineBorrows) + "; textures created " + std::to_string(s.textureCreates) +
               "; pools " + std::to_string(s.poolTextures) + " texture(s), " +
               std::to_string(s.poolBytes / 1024u) + " KiB, attachment sets " +
               std::to_string(s.attachmentSets) + ", idle surface textures " +
               std::to_string(s.surfaceTexturesIdle) + ", pool frees " + std::to_string(s.poolFrees) +
               "; after prewarm: shader compiles " + std::to_string(s.shaderCompiles) +
               ", pipeline creates " + std::to_string(s.pipelineCreates) + "; transient pool protected " +
               std::to_string(s.poolProtectedBytes / 1024u) + " KiB, expired " +
               std::to_string(s.poolExpiredBytes / 1024u) + " KiB, budget evictions " +
               std::to_string(s.poolBudgetEvictions) + ", age evictions " +
               std::to_string(s.poolAgeEvictions));
}

/// T-656: the counters of xt_gpu_target_memory_stats (50 §50.ab item 1), all fields.
xt_gpu_memory_stats collectMemoryStats(const xt_gpu_target& t) {
    xt_gpu_memory_stats m;
    std::memset(&m, 0, sizeof m);
    const xt::wgpu::WgpuSharedStats s = t.shared.stats();
    m.image_textures = s.imageTextures;
    m.image_bytes = s.imageBytes;
    m.image_idle_textures = s.imageIdleTextures;
    m.image_idle_bytes = s.imageIdleBytes;
    m.image_evictions = s.imageEvictions;
    m.attached_surfaces = t.attached;
    // T-675 (50 §50.ac item 2): the target-level pools -- transients, released cached targets, the
    // surface attachment sets and the idle surface textures -- count as transient memory; a surface
    // counts only the texture it holds. T-678 (50 §50.ac addendum item 2): the stencil attachments in
    // there count at the backend's Stencil8 size (xt::wgpu::stencil8TexelBytes: 4 B/texel on D3D12).
    m.transient_bytes = s.poolBytes;
    for (xt::capi::SurfaceGpu* g : t.gpus) {
        if (g->texture != nullptr) {  // WGPUTextureFormat_RGBA8Unorm, one sample
            m.surface_bytes += static_cast<uint64_t>(g->width) * static_cast<uint64_t>(g->height) * 4u;
        }
        m.atlas_bytes += g->renderer.stats().coverageAtlasBytes;  // T-660b: coverage atlas pages (R8)
        if (g->layers != nullptr) {
            // T-709 (50 §50.af item 2): the layer targets count as the surface's textures, their glyph
            // atlas pages as atlas memory.
            m.surface_bytes += g->layers->builder.stats().targetBytes;
#ifdef XT_CAPI_GPU_ATLAS
            if (g->layers->atlas != nullptr) {
                const auto side = static_cast<uint64_t>(g->layers->atlas->textureSize());
                m.atlas_bytes += static_cast<uint64_t>(g->layers->atlas->textureCount()) * side * side;
            }
#endif
        }
#ifdef XT_CAPI_GPU_ATLAS
        if (g->atlas != nullptr) {
            const auto side = static_cast<uint64_t>(g->atlas->textureSize());
            m.atlas_bytes += static_cast<uint64_t>(g->atlas->textureCount()) * side * side;  // R8 pages
        }
#endif
    }
    return m;
}

/// Builds the GPU resources of `s` on `t` (its renderer on the target's device, its glyph atlas):
/// on first attach, and again after a release (T-641). Empty, or why not.
std::string buildSurfaceGpu(xt_surface& s, xt_gpu_target& t) {
    std::unique_ptr<xt::capi::SurfaceGpu, xt::capi::SurfaceGpuDeleter> gpu(new xt::capi::SurfaceGpu);
    gpu->target = &t;
    t.gpus.push_back(gpu.get());  // T-656: removed again by ~SurfaceGpu
    xt::wgpu::WgpuInit in = t.init;
    in.targetFormat = WGPUTextureFormat_RGBA8Unorm;
    // T-647 (50 §50.z addendum): at 4 the renderer draws into its own 4x colour and stencil attachments
    // and resolves into the surface texture (WgpuDevice::ensureSurfaceAttachments), which is what the
    // composer samples and readback sees -- for window and offscreen targets alike.
    in.sampleCount = t.sampleCount;
    in.shared = &t.shared;  // T-648: the target's pipelines and image textures
    in.error.clear();
    if (!gpu->renderer.init(in)) return std::string("WgpuRenderer::init: ") + gpu->renderer.lastError();
    // T-685 (50 §50.ac addendum 2): the transients this surface borrows are its own while it is shown
    // (the composer's stay the anonymous user's).
    if (xt::wgpu::WgpuDevice* d = gpu->renderer.device(); d != nullptr) d->setPoolUser(true);
#ifdef XT_CAPI_GPU_ATLAS
    if (s.owner->ftShaper != nullptr) {
        // T-648: one 1024 x 1024 page first, grown to the full 2048 x 2048 once it is full.
        gpu->atlas = std::make_unique<xt::text::GlyphAtlas>(*s.owner->ftShaper, kAtlasPage, kAtlasFirstPage);
        gpu->renderer.setGlyphAtlas(gpu->atlas.get());
    }
#endif
    s.gpu = std::move(gpu);
    return std::string();
}

/// Attaches `s` to `t` (the render path; the resources are built by buildSurfaceGpu).
void attachSurface(xt_surface& s, xt_gpu_target& t) {
    s.gpuTarget = &t;
    ++t.attached;
    s.renderPath = xt::capi::RenderPath::Gpu;
}

// -- T-688 (50 §50.ae): image preload. --

/// The image variant scale a surface of `t`'s size picks for `bundle` at device pixel ratio 1: the
/// uiscale applyUiScale (capi_surface.cpp) computes for that size, limited as Context::applyUiScale
/// limits it (clamp, then the DPI floor -- T-820: never multiplied by the dpr, 25 §25.4.2.x), times the
/// ratio 1 -- what FrameBuilder hands the resolver (dpr x the element's `ui-scale` factor, `auto` = the
/// uiscale).
float preloadScale(const xt_gpu_target& t, const xt::bundle::Bundle& bundle) {
    const xt::bundle::BundleSpec* given = bundle.spec();
    const xt::bundle::BundleSpec spec = given != nullptr ? *given : xt::bundle::BundleSpec{};
    const float base = xt::bundle::uiScaleFor(spec.uiscaleRule, static_cast<float>(t.width),
                                              static_cast<float>(t.height), spec.logicalHeight);
    const xt::bundle::UiScaleLimits& limits = spec.uiscaleLimits;
    float scale = std::min(std::max(base, limits.clampMin), limits.clampMax);
    scale = std::max(scale, limits.dpiFloor * 1.0f);
    if (!(scale > 0.0f) || !std::isfinite(scale)) scale = base;
    return scale;
}

/// One preload call: the target, the bundle, the variant scale and what was done so far.
struct Preload {
    xt_gpu_target& t;
    xt::bundle::Bundle& bundle;
    float scale = 1.0f;
    uint32_t loaded = 0;
    uint32_t found = 0;
    std::string firstFailure;  ///< the first file that could not be read or decoded
};

/// True when the manifest's `assets[]` row of `name` says it is an image.
bool manifestImage(const xt::bundle::Bundle& bundle, std::string_view name) {
    for (const xt::bundle::AssetInfo& a : bundle.assetTable()) {
        if (a.file == name) return a.kind == "image";
    }
    return false;
}

/// Puts asset `asset` of `view` into the target's shared image cache through the draw path's own lookup,
/// key and upload (xt::gpu::acquireImageTexture), then ends the use: the texture is idle, the most recent.
void preloadAsset(Preload& p, const xt::IrView& view, xt::AssetId asset, std::string_view name) {
    xt::wgpu::WgpuDevice* device = p.t.composer.device();
    xt::gpu::ImageTextureRequest r;
    r.device = device;
    r.assets = &p.bundle.assets();
    r.ir = &view;
    r.asset = asset;
    r.scale = p.scale;
    r.maxTexture = device->maxTextureSize();
    const xt::gpu::ImageTextureResult got = xt::gpu::acquireImageTexture(r);
    if (got.outcome != xt::gpu::ImageTextureResult::Outcome::Ok) {
        if (p.firstFailure.empty()) p.firstFailure = std::string(name);
        return;
    }
    // 50 §50.ae: no mip chain is made, but the levels a compressed file stores come with its upload.
    const xt::CompressedImage* c = got.compressed;
    if (got.created && c != nullptr && c->levels > 1) {
        const size_t level0 = xt::ddsLevelSize(c->format, c->width, c->height);
        if (c->blocks.size() > level0) {
            (void)device->requestMipChain(got.tex, c->blocks.data() + level0,
                                          static_cast<uint32_t>(c->levels));
        }
    }
    if (got.created) ++p.loaded;
    ++p.found;
    device->releaseSharedImage(got.tex, /*keep=*/true);
}

/// Reads `ui`'s .xtir and preloads every image entry of its asset table whose name -- an atlas cell's
/// source name for `atlas:<n>#<asset>` -- is one of `names`; `matched` is set when one was. The IR's hold
/// on the resolver's decodes ends before it is freed (AssetResolver::releaseSource).
xt_status preloadFromUi(Preload& p, const xt::bundle::UiInfo& ui, const std::vector<std::string_view>& names,
                        bool& matched, const char* function) {
    std::string bytes;
    if (!xt::capi::readBundleFile(p.bundle, ui.file, bytes)) {
        return fail(XT_STATUS_IO_ERROR, std::string(function) + ": cannot read '" + ui.file + "'");
    }
    const std::vector<uint8_t> blob(bytes.begin(), bytes.end());
    xt::Result<xt::IrView> opened = xt::IrView::open(xt::IrBlob{blob.data(), blob.size()});
    if (!opened.ok()) {
        return fail(XT_STATUS_INVALID_IR, std::string(function) + ": '" + ui.file + "' is not a valid .xtir");
    }
    const xt::IrView& view = opened.value();
    const std::span<const xt::IrAsset> table = view.assets().assets();
    for (size_t i = 0; i < table.size(); ++i) {
        if (table[i].kind != 0 || table[i].embedded != 0) continue;  // images named by file only
        const std::string_view source = xt::atlasAssetSource(view.string(table[i].name));
        if (std::find(names.begin(), names.end(), source) == names.end()) continue;
        matched = true;
        preloadAsset(p, view, static_cast<xt::AssetId>(i), source);
    }
    p.bundle.assets().releaseSource(view);
    return XT_STATUS_OK;
}

/// The common checks and the end of both preload functions: the arguments, then `body`, then
/// `out_loaded`, the first failure in last_error and an info line.
template <class Body>
xt_status runPreload(const char* function, xt_gpu_target* t, xt_bundle* b, const char* name,
                     uint32_t* outLoaded, Body&& body) {
    if (outLoaded != nullptr) *outLoaded = 0;
    if (t == nullptr || b == nullptr || b->bundle == nullptr || name == nullptr) {
        return fail(XT_STATUS_INVALID_ARGUMENT, std::string(function) + ": target, bundle or name is NULL");
    }
    if (t->composer.device() == nullptr) {
        return fail(XT_STATUS_INTERNAL, std::string(function) + ": the target has no device");
    }
    Preload p{*t, *b->bundle, 1.0f, 0, 0, {}};
    p.scale = preloadScale(*t, *b->bundle);
    const xt_status status = body(p);
    if (status != XT_STATUS_OK) return status;
    if (outLoaded != nullptr) *outLoaded = p.loaded;
    logSharedCache(*t);
    xt::capi::log(0, std::string("xt_gpu_target: preloaded ") + name + ": " + std::to_string(p.loaded) +
                         " image texture(s) uploaded, " + std::to_string(p.found - p.loaded) +
                         " already cached" +
                         (p.firstFailure.empty() ? "" : "; skipped " + p.firstFailure + " (not readable)"));
    if (!p.firstFailure.empty()) {
        xt::capi::setLastError(std::string(function) + ": '" + p.firstFailure +
                               "' could not be read or decoded; skipped");
    }
    return XT_STATUS_OK;
}

// -- T-709 (50 §50.af): world layers. --

/// The world-layer state of `s`, made (with its builder bound to the surface renderer's device and an
/// atlas of its own) when it has none. Null when the renderer has no device.
xt::capi::SurfaceLayers* ensureLayers(xt_surface& s) {
    xt::capi::SurfaceGpu& g = *s.gpu;
    if (g.layers != nullptr) return g.layers.get();
    xt::wgpu::WgpuDevice* device = g.renderer.device();
    if (device == nullptr) return nullptr;
    g.layers = std::make_unique<xt::capi::SurfaceLayers>();
    g.layers->builder.init(*device);
#ifdef XT_CAPI_GPU_ATLAS
    if (s.owner->ftShaper != nullptr) {
        // The layers' own atlas (WorldLayerBuilder::setGlyphAtlas: one no other FrameBuilder uses), sized
        // as the surface's.
        g.layers->atlas =
            std::make_unique<xt::text::GlyphAtlas>(*s.owner->ftShaper, kAtlasPage, kAtlasFirstPage);
        g.layers->builder.setGlyphAtlas(g.layers->atlas.get());
    }
#endif
    return g.layers.get();
}

/// Drops the composer entries whose layer target the build just made may have replaced: all of them when
/// the build acquired a target (a new texture can reuse a released one's handle), else those whose layer
/// is gone or names another texture now.
void syncLayerBindings(xt::capi::SurfaceGpu& g) {
    xt::capi::SurfaceLayers& L = *g.layers;
    if (L.bound.empty()) return;
    if (L.builder.stats().targetsCreated > 0 || !L.valid) {
        g.unbindLayers();
        return;
    }
    xt::wgpu::WgpuDevice* device = g.renderer.device();
    const std::span<const xt::gpu::WorldLayer> layers = L.builder.layers();
    for (size_t i = 0; i < L.bound.size();) {
        const xt::capi::SurfaceLayers::Bound& b = L.bound[i];
        const bool alive = std::any_of(layers.begin(), layers.end(), [&](const xt::gpu::WorldLayer& l) {
            return l.texture == b.layer && l.texWidth == b.width && l.texHeight == b.height &&
                   device != nullptr && device->texture(l.texture) == b.texture;
        });
        if (alive) {
            ++i;
            continue;
        }
        if (b.composer != 0 && g.target->composer.device() != nullptr) {
            g.target->composer.device()->unregisterExternalTexture(b.composer);
        }
        if (b.view != nullptr) wgpuTextureViewRelease(b.view);
        L.bound.erase(L.bound.begin() + static_cast<std::ptrdiff_t>(i));
    }
}

/// The composer's handle of `layer`'s target (registered on first use). 0 when it cannot be had.
xt::gpu::TextureHandle composerHandle(xt::capi::SurfaceGpu& g, const xt::gpu::WorldLayer& layer) {
    xt::capi::SurfaceLayers& L = *g.layers;
    xt::wgpu::WgpuDevice* device = g.renderer.device();
    xt::wgpu::WgpuDevice* composer = g.target->composer.device();
    if (device == nullptr || composer == nullptr) return 0;
    WGPUTexture texture = device->texture(layer.texture);
    if (texture == nullptr) return 0;
    for (const xt::capi::SurfaceLayers::Bound& b : L.bound) {
        if (b.layer == layer.texture && b.texture == texture) return b.composer;
    }
    xt::capi::SurfaceLayers::Bound b;
    b.layer = layer.texture;
    b.texture = texture;
    b.width = layer.texWidth;
    b.height = layer.texHeight;
    b.view = wgpuTextureCreateView(texture, nullptr);
    if (b.view == nullptr) return 0;
    b.composer = composer->registerExternalTexture(b.view, layer.texWidth, layer.texHeight);
    if (b.composer == 0) {
        wgpuTextureViewRelease(b.view);
        return 0;
    }
    L.bound.push_back(b);
    return b.composer;
}

/// The layers of `s`'s last render_layers_gpu (empty when it has none).
std::span<const xt::gpu::WorldLayer> surfaceLayers(const xt_surface& s) {
    if (s.gpu == nullptr || s.gpu->layers == nullptr || !s.gpu->layers->valid) return {};
    return s.gpu->layers->builder.layers();
}

}  // namespace

extern "C" {

XT_CAPI xt_gpu_target* xt_gpu_target_create(const xt_gpu_target_desc* desc) {
    XT_CAPI_GUARD_OR("xt_gpu_target_create", xt_gpu_target*, nullptr, {
        auto failNull = [](const std::string& message) -> xt_gpu_target* {
            xt::capi::setLastError("xt_gpu_target_create: " + message);
            return nullptr;
        };
        if (desc == nullptr) return failNull("desc is NULL");
        // T-647: sample_count was appended; a desc ending before it (an older host) still creates.
        // T-666 (10 §10.11 item 8): 0 -- set, or not covered by struct_size -- means 1 (it meant 4).
        constexpr size_t kMinDescSize = offsetof(xt_gpu_target_desc, sample_count);
        if (desc->struct_size < kMinDescSize) {
            return failNull("desc->struct_size is smaller than sizeof(xt_gpu_target_desc)");
        }
        const int samples = desc->struct_size >= sizeof(xt_gpu_target_desc) ? desc->sample_count : 0;
        if (samples != 0 && samples != 1 && samples != 4) {
            return failNull("sample_count " + std::to_string(samples) + " (allowed: 0 = 1, 1, 4)");
        }
        if (desc->width <= 0 || desc->height <= 0 || desc->width > kMaxSide || desc->height > kMaxSide) {
            return failNull("width / height must be in 1.." + std::to_string(kMaxSide));
        }
        xt::wgpu::WgpuNativeWindow native;
        const xt_native_window& w = desc->window;
        switch (w.kind) {
            case XT_WINDOW_NONE:
                break;
            case XT_WINDOW_WIN32:
                native.kind = xt::wgpu::WgpuNativeWindow::Kind::Win32;
                native.window = w.u.win32.hwnd;
                break;
            case XT_WINDOW_COCOA:
                native.kind = xt::wgpu::WgpuNativeWindow::Kind::CocoaWindow;
                native.window = w.u.cocoa.ns_window;
                break;
            case XT_WINDOW_X11:
                native.kind = xt::wgpu::WgpuNativeWindow::Kind::X11;
                native.display = w.u.x11.display;
                native.x11Window = w.u.x11.window;
                break;
            case XT_WINDOW_WAYLAND:
                native.kind = xt::wgpu::WgpuNativeWindow::Kind::Wayland;
                native.display = w.u.wayland.display;
                native.window = w.u.wayland.surface;
                break;
            default:
                return failNull("unknown window kind " + std::to_string(w.kind));
        }
        auto t = std::make_unique<xt_gpu_target>();
        t->width = desc->width;
        t->height = desc->height;
        t->sampleCount = samples == 4 ? 4u : 1u;
        xt::wgpu::WgpuHeadlessOptions options;
        options.backend = desc->backend != nullptr ? desc->backend : "";
        t->init = xt::wgpu::WgpuInit::headless(options);
        if (t->init.device == nullptr) return failNull(t->init.error);
        // T-671 (50 §50.ab addendum 2, items 1 and 5): the adapter and the memory strategy in effect.
        for (size_t at = 0; at < t->init.notes.size();) {
            const size_t nl = std::min(t->init.notes.find('\n', at), t->init.notes.size());
            xt::capi::log(0, "xt_gpu_target: " + t->init.notes.substr(at, nl - at));
            at = nl + 1;
        }
        if (native.kind != xt::wgpu::WgpuNativeWindow::Kind::None) {
            std::string error;
            // T-749 (10 §10.10.2, T-758 supplement): the composer only draws textured quads over target 0
            // and never samples it, so the surface is drawn into directly even without TEXTURE_BINDING.
            if (!t->window.create(t->init, native, desc->vsync != 0, error, xt::wgpu::SurfaceReads::None)) {
                return failNull(error);
            }
            t->format = t->window.format();
            t->window.configure(t->init.device, t->width, t->height);
            xt::capi::log(0, "xt_gpu_target: " + t->window.usageNote());  // T-758
        } else {
            t->format = WGPUTextureFormat_RGBA8Unorm;
            if (!t->createOffscreen()) return failNull("the offscreen texture cannot be created");
        }
        xt::wgpu::WgpuInit in = t->init;
        in.targetFormat = t->format;
        in.sampleCount = 1;      // one axis-aligned textured quad per surface: nothing to antialias
        in.shared = &t->shared;  // T-648: the composer makes the target's pipeline set
        in.error.clear();
        if (!t->composer.init(in))
            return failNull(std::string("WgpuRenderer::init: ") + t->composer.lastError());
        t->composer.resize(t->width, t->height);
        // T-683 (50 §50.ad item 2): every built-in program compiled now, for the surfaces' renderers
        // (RGBA8, the target's sample count), so no first show waits for a shader compile.
        const xt::wgpu::WgpuPipelinePrewarm warm =
            t->composer.device()->prewarmPipelines(WGPUTextureFormat_RGBA8Unorm, t->sampleCount);
        char ms[32];
        std::snprintf(ms, sizeof ms, "%.1f", warm.ms);
        xt::capi::log(warm.ok ? 0 : 1,
                      "xt_gpu_target: prewarmed " + std::to_string(warm.shaders) + " shader(s), " +
                          std::to_string(warm.pipelines) + " pipeline(s) in " + ms + " ms" +
                          (warm.ok ? "" : " (a pipeline failed; its draws will be skipped)"));
        return t.release();
    });
}

XT_CAPI xt_status xt_gpu_target_destroy(xt_gpu_target* t) {
    XT_CAPI_GUARD("xt_gpu_target_destroy", {
        if (t == nullptr) return XT_STATUS_OK;
        if (t->attached > 0) {
            return fail(XT_STATUS_BUSY, "xt_gpu_target_destroy: " + std::to_string(t->attached) +
                                            " surface(s) still attached; destroy them or call "
                                            "xt_surface_detach_gpu first");
        }
        t->inflight.waitIdle(t->init.device);  // T-674: the GPU is done with everything before it is freed
        const uint32_t images = t->shared.stats().imageTextures;
        // T-674: the frames-in-flight counters and the target's submit / pass totals, for tests and tools.
        char waitMs[32];
        std::snprintf(waitMs, sizeof waitMs, "%.3f", t->inflight.waitMs());
        const std::string inflight =
            "; frames in flight: " + std::to_string(t->inflight.waits()) + " wait(s), " + waitMs +
            " ms, peak " + std::to_string(t->inflight.peakInFlight()) + "; " + std::to_string(t->submits) +
            " submits, " + std::to_string(t->passes) + " passes";
        delete t;
        xt::capi::log(0, "xt_gpu_target: destroyed; " + std::to_string(images) +
                             " shared image texture(s) freed" + inflight);
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_resize(xt_gpu_target* t, int width, int height) {
    XT_CAPI_GUARD("xt_gpu_target_resize", {
        if (t == nullptr) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_resize: target is NULL");
        if (width <= 0 || height <= 0 || width > kMaxSide || height > kMaxSide) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_gpu_target_resize: width / height must be in 1.." + std::to_string(kMaxSide));
        }
        if (t->inFrame) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_resize: called inside a frame");
        }
        if (width == t->width && height == t->height) return XT_STATUS_OK;
        t->width = width;
        t->height = height;
        if (t->windowed()) {
            t->window.configure(t->init.device, width, height);
        } else if (!t->createOffscreen()) {
            return fail(XT_STATUS_INTERNAL, "xt_gpu_target_resize: the offscreen texture cannot be created");
        }
        t->composer.resize(width, height);
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_begin_frame(xt_gpu_target* t, const float clear_rgba[4]) {
    XT_CAPI_GUARD("xt_gpu_target_begin_frame", {
        if (t == nullptr)
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_begin_frame: target is NULL");
        if (t->inFrame) {
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_gpu_target_begin_frame: the previous frame was not ended (xt_gpu_target_end_frame)");
        }
        // T-674 (50 §50.ac item 1): waits only while the GPU is more than two frames behind.
        t->inflight.beginFrame(t->init.device, t->init.queue);
        const std::string inflightWarning = t->inflight.takeWarning();
        if (!inflightWarning.empty()) xt::capi::log(1, "xt_gpu_target_begin_frame: " + inflightWarning);
        t->inFrame = true;
        static std::atomic<uint64_t> frameSerials{0};  // T-709: unique across targets and threads
        t->frameSerial = frameSerials.fetch_add(1) + 1;
        if (t->windowed()) {
            if (t->window.acquire()) {  // else: skipped (minimized, timeout)
                t->frameTexture = t->window.frameTexture();
                t->frameView = t->window.frameView();
            }
        } else {
            t->frameTexture = t->offscreen;
            t->frameView = wgpuTextureCreateView(t->offscreen, nullptr);
        }
        // T-658: submitted with the first composite of the frame, or by end_frame / readback.
        t->pendingClear = clearColor(clear_rgba);
        t->clearPending = t->frameView != nullptr;
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_end_frame(xt_gpu_target* t) {
    XT_CAPI_GUARD("xt_gpu_target_end_frame", {
        if (t == nullptr) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_end_frame: target is NULL");
        if (!t->inFrame) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_end_frame: no frame was begun");
        }
        const std::string error = flushPendingClear(*t);  // T-658: nothing was composited
        if (t->windowed()) {
            t->window.present();  // releases the frame's texture
            t->frameView = nullptr;
            t->frameTexture = nullptr;
        } else {
            t->releaseFrame();  // the work is already submitted
        }
        t->inFrame = false;
        t->inflight.endFrame(t->init.queue);         // T-674: the frame's last submission index
        wgpuDevicePoll(t->init.device, 0, nullptr);  // let finished work be reclaimed, never waits
        t->shared.endFrame();                        // T-675: the target-level pools age by these frames
        logSharedCache(*t);
        if (!error.empty()) return fail(XT_STATUS_INTERNAL, "xt_gpu_target_end_frame: " + error);
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_readback(xt_gpu_target* t, void* rgba, int stride) {
    XT_CAPI_GUARD("xt_gpu_target_readback", {
        if (t == nullptr) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_readback: target is NULL");
        const size_t rowBytes = static_cast<size_t>(t->width) * 4u;
        if (rgba == nullptr || stride < 0 || static_cast<size_t>(stride) < rowBytes) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_gpu_target_readback: rgba is NULL or stride < width * 4");
        }
        WGPUTexture texture = t->windowed() ? t->frameTexture : t->offscreen;
        if (texture == nullptr) {
            return fail(
                XT_STATUS_UNSUPPORTED,
                "xt_gpu_target_readback: a window target is read only between begin_frame and end_frame");
        }
        if (t->windowed() && !t->window.frameCopyable()) {
            return fail(XT_STATUS_UNSUPPORTED,
                        "xt_gpu_target_readback: this window's swapchain cannot be copied");
        }
        // T-658: inside a frame whose clear no composite has carried yet, the clear comes first.
        const std::string clearError = flushPendingClear(*t);
        if (!clearError.empty()) {
            return fail(XT_STATUS_INTERNAL, "xt_gpu_target_readback: " + clearError);
        }
        std::vector<uint8_t> tight(rowBytes * static_cast<size_t>(t->height));
        if (!t->composer.readback(texture, std::span<uint8_t>(tight))) {
            return fail(XT_STATUS_INTERNAL,
                        std::string("xt_gpu_target_readback: ") + t->composer.lastError());
        }
        auto* out = static_cast<uint8_t*>(rgba);
        for (int y = 0; y < t->height; ++y) {
            std::memcpy(out + static_cast<size_t>(y) * static_cast<size_t>(stride),
                        tight.data() + rowBytes * static_cast<size_t>(y), rowBytes);
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_render_gpu(xt_surface* s, xt_gpu_target* t, const xt_rect* dest) {
    XT_CAPI_GUARD("xt_surface_render_gpu", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_render_gpu");
        if (t == nullptr) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_render_gpu: target is NULL");
        if (s->worldLayers) {  // T-709 (50 §50.af item 1): no single picture in world-layer output
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_surface_render_gpu: the surface is in world-layer output; use "
                "xt_surface_render_layers_gpu, or xt_surface_set_world_layers(s, 0) for a flat picture");
        }
        if (s->renderPath == xt::capi::RenderPath::Software) {
            return fail(XT_STATUS_UNSUPPORTED,
                        "xt_surface_render_gpu: the surface renders with xt_surface_render; call "
                        "xt_surface_detach_gpu first");
        }
        if (s->gpuTarget != nullptr && s->gpuTarget != t) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_render_gpu: the surface is attached to another target; call "
                        "xt_surface_detach_gpu first");
        }
        if (!t->inFrame) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_render_gpu: called outside xt_gpu_target_begin_frame / end_frame");
        }
        const xt_rect rect = dest != nullptr ? *dest : xt_rect{0, 0, t->width, t->height};
        if (rect.w <= 0 || rect.h <= 0) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_render_gpu: dest has no area");
        }
        if (s->width > kMaxSide || s->height > kMaxSide) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_render_gpu: the surface is larger than " +
                                                        std::to_string(kMaxSide) + " px");
        }
        if ((s->gpu == nullptr || s->gpu->texture == nullptr) && s->gpuTarget != nullptr &&
            !s->ctx->isVisible(s->handle)) {
            // T-641 (50 §50.aa item 4): released while hidden; a hidden surface paints nothing, so
            // its resources are rebuilt only once it has something to draw again. T-675 (50 §50.ac
            // item 2): a hidden surface that gave its texture back (trim-on-hide) is not built either --
            // an empty build would drop the gradient tables it keeps for the next show.
            xt::capi::DispatchScope scope(*s);
            s->ctx->render(s->list, &s->damage);
            if (s->list.commands.empty()) return XT_STATUS_OK;
        }
        if (s->gpu == nullptr) {
            const std::string error = buildSurfaceGpu(*s, *t);
            if (!error.empty()) return fail(XT_STATUS_INTERNAL, "xt_surface_render_gpu: " + error);
            if (s->gpuTarget == nullptr) {
                attachSurface(*s, *t);
            } else {
                xt::capi::log(0, "xt_surface_render_gpu: GPU resources rebuilt (" + s->uiId + ")");
            }
        }
        xt::capi::SurfaceGpu& g = *s->gpu;
        const bool resized = g.ensureRenderSize(s->width, s->height);
        {
            xt::capi::DispatchScope scope(*s);
            s->ctx->render(s->list, &s->damage);
            // The texture keeps the last frame: an unchanged frame is not drawn again. Any change redraws
            // the whole texture (cleared), so the partial-damage path of the builder is not needed.
            // T-662: a frame whose budget held a coverage shape back is drawn again until none is.
            const bool unchanged =
                g.built && !resized && !g.repaint && !s->damage.full && s->damage.rects.empty();
            if (!unchanged) {
                xt::PerfTracker* perf = &s->ctx->perf();
                const uint32_t draws = g.renderer.build(s->list, s->owner->shaper.get(),
                                                        &s->owner->bundle->assets(), nullptr, perf, nullptr);
                const bool wasEmpty = g.built && g.empty;
                g.built = true;
                g.repaint = g.renderer.wantsAnotherFrame();
                g.empty = draws == 0;
                if (g.empty) {
                    // T-658 (50 §50.ab item 3 ii / iii): nothing to draw -- no texture, no 4x attachments,
                    // no submit and no composite until a frame has something.
                    g.executed = g.skipped = 0;
                    if (!wasEmpty) {
                        xt::capi::log(0, "xt_surface_render_gpu: " + s->uiId +
                                             " has nothing to draw; not composited (" +
                                             (g.texture != nullptr ? "texture kept" : "no texture") +
                                             "; target " + std::to_string(t->submits) + " submits, " +
                                             std::to_string(t->passes) + " passes)");
                    }
                } else {
                    if (wasEmpty) {
                        xt::capi::log(0, "xt_surface_render_gpu: " + s->uiId + " draws again (target " +
                                             std::to_string(t->submits) + " submits, " +
                                             std::to_string(t->passes) + " passes)");
                    }
                    bool fresh = false;
                    if (!g.ensureTexture(s->width, s->height, fresh)) {
                        g.built = false;  // drawn by the next call
                        return fail(XT_STATUS_INTERNAL,
                                    "xt_surface_render_gpu: the surface texture cannot be created");
                    }
                    g.renderer.setClearColor(0, 0, 0, 0);
                    g.renderer.submit(g.view, perf);
                    const xt::wgpu::WgpuStats stats = g.renderer.stats();
                    g.executed = stats.drawCalls;
                    g.skipped = stats.skippedCmds;
                    if (xt::wgpu::WgpuDevice* d = g.renderer.device();
                        d != nullptr && d->lastSubmissionIndex() != 0) {
                        t->inflight.noteSubmit(d->lastSubmissionIndex());  // T-674
                    }
                    ++t->submits;
                    t->passes += stats.passes;
                }
                logSharedCache(*t);
                if (g.renderer.lastError()[0] != '\0') {
                    xt::capi::log(2, std::string("xt_surface_render_gpu: ") + g.renderer.lastError());
                }
            }
        }
        if (!g.empty && t->frameView != nullptr && buildCompositeFrame(*t, g.handle, rect)) {
            const std::string error = submitComposite(*t);
            if (!error.empty()) return fail(XT_STATUS_INTERNAL, "xt_surface_render_gpu: " + error);
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_detach_gpu(xt_surface* s) {
    XT_CAPI_GUARD("xt_surface_detach_gpu", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_detach_gpu");
        xt::capi::detachGpu(*s);
        return XT_STATUS_OK;
    });
}

// -- T-656 (50 §50.ab item 1): the target's image budget, trim and memory counters. --

XT_CAPI xt_status xt_gpu_target_set_image_budget(xt_gpu_target* t, uint64_t idle_bytes) {
    XT_CAPI_GUARD("xt_gpu_target_set_image_budget", {
        if (t == nullptr)
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_set_image_budget: target is NULL");
        t->shared.setImageIdleBudget(idle_bytes);
        logSharedCache(*t);
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_trim(xt_gpu_target* t) {
    XT_CAPI_GUARD("xt_gpu_target_trim", {
        if (t == nullptr) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_trim: target is NULL");
        if (t->inFrame) return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_trim: called inside a frame");
        t->inflight.waitIdle(t->init.device);  // T-674: the GPU is done with everything before it is freed
        const uint32_t freed = t->shared.trimIdleImages();
        // T-675 (50 §50.ac item 2): and every pooled texture no surface borrows (none is borrowed
        // outside a frame).
        const uint32_t pooled = t->shared.trimPools();
        wgpuDevicePoll(t->init.device, 0, nullptr);  // let the driver reclaim it, never waits
        logSharedCache(*t);
        xt::capi::log(0, "xt_gpu_target: trimmed; " + std::to_string(freed) +
                             " idle image texture(s) freed, " + std::to_string(pooled) +
                             " pooled texture(s) freed");
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_gpu_target_memory_stats(const xt_gpu_target* t, xt_gpu_memory_stats* out) {
    XT_CAPI_GUARD("xt_gpu_target_memory_stats", {
        if (t == nullptr || out == nullptr) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_gpu_target_memory_stats: target or out is NULL");
        }
        const xt_gpu_memory_stats m = collectMemoryStats(*t);
        // Only the fields `struct_size` covers (struct_size itself is the caller's).
        const size_t size = std::min<size_t>(out->struct_size, sizeof m);
        if (size > sizeof m.struct_size) {
            std::memcpy(reinterpret_cast<uint8_t*>(out) + sizeof m.struct_size,
                        reinterpret_cast<const uint8_t*>(&m) + sizeof m.struct_size,
                        size - sizeof m.struct_size);
        }
        return XT_STATUS_OK;
    });
}

// -- T-688 (50 §50.ae): image preload. --

XT_CAPI xt_status xt_gpu_target_preload_ui(xt_gpu_target* t, xt_bundle* b, const char* ui_id,
                                           uint32_t* out_loaded) {
    XT_CAPI_GUARD("xt_gpu_target_preload_ui", {
        return runPreload("xt_gpu_target_preload_ui", t, b, ui_id, out_loaded, [&](Preload& p) -> xt_status {
            const xt::bundle::UiInfo* ui = p.bundle.ui(ui_id);
            if (ui == nullptr) {
                return fail(XT_STATUS_NOT_FOUND,
                            std::string("xt_gpu_target_preload_ui: the bundle has no UI '") + ui_id + "'");
            }
            std::vector<std::string_view> names;
            for (const std::string& a : ui->assets) {
                if (manifestImage(p.bundle, a)) names.push_back(a);
            }
            if (names.empty()) return XT_STATUS_OK;
            bool matched = false;
            return preloadFromUi(p, *ui, names, matched, "xt_gpu_target_preload_ui");
        });
    });
}

XT_CAPI xt_status xt_gpu_target_preload_image(xt_gpu_target* t, xt_bundle* b, const char* asset,
                                              uint32_t* out_loaded) {
    XT_CAPI_GUARD("xt_gpu_target_preload_image", {
        return runPreload(
            "xt_gpu_target_preload_image", t, b, asset, out_loaded, [&](Preload& p) -> xt_status {
                const std::string_view name(asset);
                if (!manifestImage(p.bundle, name)) {
                    return fail(XT_STATUS_NOT_FOUND, std::string("xt_gpu_target_preload_image: '") + asset +
                                                         "' is not an image asset of the bundle");
                }
                // The first UI (manifest order) that lists it and whose asset table names it.
                const std::vector<std::string_view> names{name};
                for (const xt::bundle::UiInfo& ui : p.bundle.uis()) {
                    if (std::find(ui.assets.begin(), ui.assets.end(), name) == ui.assets.end()) continue;
                    bool matched = false;
                    const xt_status status =
                        preloadFromUi(p, ui, names, matched, "xt_gpu_target_preload_image");
                    if (status != XT_STATUS_OK || matched) return status;
                }
                return fail(
                    XT_STATUS_NOT_FOUND,
                    std::string("xt_gpu_target_preload_image: no UI of the bundle uses '") + asset + "'");
            });
    });
}

// -- T-709 (50 §50.af): world layers. --

XT_CAPI xt_status xt_surface_set_world_layers(xt_surface* s, int enabled) {
    XT_CAPI_GUARD("xt_surface_set_world_layers", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_set_world_layers");
        const bool on = enabled != 0;
        s->ctx->setWorldLayerOutput(on);
        s->worldLayers = on;
        if (!on) {
            if (s->gpu != nullptr) s->gpu->releaseLayers();  // off: the layer targets go
            s->layerFrame = 0;
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_set_world_layer_density(xt_surface* s, float density) {
    XT_CAPI_GUARD("xt_surface_set_world_layer_density", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_set_world_layer_density");
        if (!std::isfinite(density) || !(density > 0.0f)) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_set_world_layer_density: density must be > 0");
        }
        s->worldLayerDensity = density;
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_render_layers_gpu(xt_surface* s, xt_gpu_target* t) {
    XT_CAPI_GUARD("xt_surface_render_layers_gpu", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_render_layers_gpu");
        if (t == nullptr)
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_render_layers_gpu: target is NULL");
        if (!s->worldLayers) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_render_layers_gpu: world layers are off (xt_surface_set_world_layers)");
        }
        if (s->renderPath == xt::capi::RenderPath::Software) {
            return fail(XT_STATUS_UNSUPPORTED,
                        "xt_surface_render_layers_gpu: the surface renders with xt_surface_render; call "
                        "xt_surface_detach_gpu first");
        }
        if (s->gpuTarget != nullptr && s->gpuTarget != t) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_render_layers_gpu: the surface is attached to another target; call "
                        "xt_surface_detach_gpu first");
        }
        if (!t->inFrame) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_render_layers_gpu: called outside xt_gpu_target_begin_frame / end_frame");
        }
        if (!s->ctx->isVisible(s->handle)) {
            // A hidden surface paints nothing and has no layers; its resources are not (re)built (as
            // xt_surface_render_gpu, 50 §50.aa item 4); trim-on-hide has released its layer targets.
            xt::capi::DispatchScope scope(*s);
            s->ctx->render(s->list, &s->damage);
            if (s->gpu != nullptr && s->gpu->layers != nullptr) s->gpu->layers->valid = false;
            s->layerFrame = t->frameSerial;
            return XT_STATUS_OK;
        }
        if (s->gpu == nullptr) {
            const std::string error = buildSurfaceGpu(*s, *t);
            if (!error.empty()) return fail(XT_STATUS_INTERNAL, "xt_surface_render_layers_gpu: " + error);
            if (s->gpuTarget == nullptr) {
                attachSurface(*s, *t);
            } else {
                xt::capi::log(0, "xt_surface_render_layers_gpu: GPU resources rebuilt (" + s->uiId + ")");
            }
        }
        xt::capi::SurfaceGpu& g = *s->gpu;
        xt::capi::SurfaceLayers* L = ensureLayers(*s);
        if (L == nullptr) {
            return fail(XT_STATUS_INTERNAL,
                        "xt_surface_render_layers_gpu: the surface renderer has no device");
        }
        // 50 §50.af item 2: the density multiplies the device pixel ratio; 1 = the list's own (the builder's
        // 0).
        L->builder.setDensity(s->worldLayerDensity == 1.0f ? 0.0f : s->worldLayerDensity * s->dpr);
        xt::capi::DispatchScope scope(*s);
        s->ctx->render(s->list, &s->damage);
        xt::PerfTracker* perf = &s->ctx->perf();
        const bool built =
            L->builder.build(s->list, s->owner->shaper.get(), &s->owner->bundle->assets(), nullptr, perf);
        L->valid = built;
        s->layerFrame = t->frameSerial;
        if (!built) {
            syncLayerBindings(g);
            switch (L->builder.status()) {
                case xt::gpu::WorldLayerBuilder::BuildStatus::NoLayers:
                    // Nothing painted in world-layer output this frame: no layers.
                    return XT_STATUS_OK;
                case xt::gpu::WorldLayerBuilder::BuildStatus::Unsupported:
                    return fail(XT_STATUS_UNSUPPORTED,
                                "xt_surface_render_layers_gpu: the device has no world-layer targets");
                default:
                    return fail(XT_STATUS_INTERNAL,
                                std::string("xt_surface_render_layers_gpu: ") + L->builder.lastError());
            }
        }
        syncLayerBindings(g);
        xt::wgpu::WgpuDevice* device = g.renderer.device();
        const WGPUSubmissionIndex before = device->lastSubmissionIndex();
        L->builder.submit();
        const xt::gpu::WorldLayerBuilder::Stats stats = L->builder.stats();
        if (stats.submitted > 0) {
            if (device->lastSubmissionIndex() != 0 && device->lastSubmissionIndex() != before) {
                t->inflight.noteSubmit(device->lastSubmissionIndex());  // T-674
            }
            t->submits += stats.submitted;
            t->passes += device->lastStats().passes;
        }
        g.executed = stats.drawCommands;
        g.skipped = 0;
        logSharedCache(*t);
        if (L->builder.lastError()[0] != '\0') {
            xt::capi::log(1, std::string("xt_surface_render_layers_gpu: ") + L->builder.lastError());
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI uint32_t xt_surface_world_layer_count(const xt_surface* s) {
    XT_CAPI_GUARD_OR("xt_surface_world_layer_count", uint32_t, 0u,
                     { return s == nullptr ? 0u : static_cast<uint32_t>(surfaceLayers(*s).size()); });
}

XT_CAPI xt_status xt_surface_world_layer_info(const xt_surface* s, uint32_t index, xt_world_layer* out) {
    XT_CAPI_GUARD("xt_surface_world_layer_info", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_world_layer_info");
        if (out == nullptr)
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_world_layer_info: out is NULL");
        const std::span<const xt::gpu::WorldLayer> layers = surfaceLayers(*s);
        if (index >= layers.size()) {
            return fail(XT_STATUS_OUT_OF_RANGE, "xt_surface_world_layer_info: index " +
                                                    std::to_string(index) + " >= count " +
                                                    std::to_string(layers.size()));
        }
        const xt::gpu::WorldLayer& l = layers[index];
        xt_world_layer w;
        std::memset(&w, 0, sizeof w);
        w.id = l.id;
        w.order = l.order;
        w.tex_width = l.texWidth;
        w.tex_height = l.texHeight;
        std::memcpy(w.world3, l.world3.m, sizeof w.world3);
        w.rect[0] = l.rect.x;
        w.rect[1] = l.rect.y;
        w.rect[2] = l.rect.w;
        w.rect[3] = l.rect.h;
        w.opacity = l.opacity;
        w.backface_hidden = l.backfaceHidden;
        w.changed = l.changed;
        // Only the fields `struct_size` covers (struct_size itself is the caller's).
        const size_t size = std::min<size_t>(out->struct_size, sizeof w);
        if (size > sizeof w.struct_size) {
            std::memcpy(reinterpret_cast<uint8_t*>(out) + sizeof w.struct_size,
                        reinterpret_cast<const uint8_t*>(&w) + sizeof w.struct_size,
                        size - sizeof w.struct_size);
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_world_layer_readback(xt_surface* s, uint32_t index, void* rgba, int stride) {
    XT_CAPI_GUARD("xt_surface_world_layer_readback", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_world_layer_readback");
        const std::span<const xt::gpu::WorldLayer> layers = surfaceLayers(*s);
        if (index >= layers.size()) {
            return fail(XT_STATUS_OUT_OF_RANGE, "xt_surface_world_layer_readback: index " +
                                                    std::to_string(index) + " >= count " +
                                                    std::to_string(layers.size()));
        }
        const xt::gpu::WorldLayer& l = layers[index];
        const size_t rowBytes = static_cast<size_t>(l.texWidth) * 4u;
        if (rgba == nullptr || stride < 0 || static_cast<size_t>(stride) < rowBytes) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_world_layer_readback: rgba is NULL or stride < tex_width * 4");
        }
        xt_gpu_target* t = s->gpuTarget;
        if (t == nullptr) return fail(XT_STATUS_INTERNAL, "xt_surface_world_layer_readback: not attached");
        if (t->inFrame && s->layerFrame == t->frameSerial) {
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_surface_world_layer_readback: call it after the xt_gpu_target_end_frame of the frame "
                "that rendered the layers");
        }
        xt::wgpu::WgpuDevice* device = s->gpu->renderer.device();
        WGPUTexture texture = device != nullptr ? device->texture(l.texture) : nullptr;
        if (texture == nullptr) {
            return fail(XT_STATUS_INTERNAL, "xt_surface_world_layer_readback: the layer has no texture");
        }
        std::vector<uint8_t> tight(rowBytes * static_cast<size_t>(l.texHeight));
        if (!t->composer.readback(texture, std::span<uint8_t>(tight))) {
            return fail(XT_STATUS_INTERNAL,
                        std::string("xt_surface_world_layer_readback: ") + t->composer.lastError());
        }
        auto* out = static_cast<uint8_t*>(rgba);
        for (uint32_t y = 0; y < l.texHeight; ++y) {
            std::memcpy(out + static_cast<size_t>(y) * static_cast<size_t>(stride),
                        tight.data() + rowBytes * y, rowBytes);
        }
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_draw_layers_gpu(xt_surface* s, xt_gpu_target* t, const xt_rect* dest,
                                             const float box_to_dest[16], float depth_scale) {
    XT_CAPI_GUARD("xt_surface_draw_layers_gpu", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_draw_layers_gpu");
        if (t == nullptr || box_to_dest == nullptr) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_draw_layers_gpu: target or box_to_dest is NULL");
        }
        if (!std::isfinite(depth_scale) || depth_scale < 0.0f) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_draw_layers_gpu: depth_scale must be >= 0 and finite");
        }
        for (int i = 0; i < 16; ++i) {
            if (!std::isfinite(box_to_dest[i])) {
                return fail(XT_STATUS_INVALID_ARGUMENT,
                            "xt_surface_draw_layers_gpu: box_to_dest is not finite");
            }
        }
        if (!s->worldLayers) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_draw_layers_gpu: world layers are off (xt_surface_set_world_layers)");
        }
        if (!t->inFrame) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_draw_layers_gpu: called outside xt_gpu_target_begin_frame / end_frame");
        }
        if ((s->gpuTarget != nullptr && s->gpuTarget != t) || s->layerFrame != t->frameSerial) {
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_surface_draw_layers_gpu: call xt_surface_render_layers_gpu on this target in this "
                "frame first");
        }
        const xt_rect rect = dest != nullptr ? *dest : xt_rect{0, 0, t->width, t->height};
        if (rect.w <= 0 || rect.h <= 0) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_draw_layers_gpu: dest has no area");
        }
        const std::span<const xt::gpu::WorldLayer> layers = surfaceLayers(*s);
        if (layers.empty() || t->frameView == nullptr) return XT_STATUS_OK;  // nothing, or a skipped frame
        const int cx0 = std::clamp(rect.x, 0, t->width), cy0 = std::clamp(rect.y, 0, t->height);
        const int cx1 =
            static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(rect.x) + rect.w, 0, t->width));
        const int cy1 =
            static_cast<int>(std::clamp<int64_t>(static_cast<int64_t>(rect.y) + rect.h, 0, t->height));
        if (cx1 <= cx0 || cy1 <= cy0) return XT_STATUS_OK;  // entirely outside the target
        // dest's offset x box_to_dest x scale(1, 1, depth_scale): root box -> homogeneous target pixels.
        xt::Mat4 m;
        std::memcpy(m.m, box_to_dest, sizeof m.m);
        for (int r = 0; r < 4; ++r) m.m[r * 4 + 2] *= depth_scale;
        for (int c = 0; c < 4; ++c) {
            m.m[c] += static_cast<float>(rect.x) * m.m[12 + c];
            m.m[4 + c] += static_cast<float>(rect.y) * m.m[12 + c];
        }
        xt::capi::SurfaceGpu& g = *s->gpu;
        xt::capi::SurfaceLayers& L = *g.layers;
        L.quadLayers.assign(layers.begin(), layers.end());
        for (xt::gpu::WorldLayer& l : L.quadLayers) l.texture = composerHandle(g, l);  // 0: left out
        xt::gpu::buildWorldLayerQuads(L.quadLayers, m, t->width, t->height, /*clear=*/false, t->frame);
        if (t->frame.cmds.empty()) return XT_STATUS_OK;
        for (xt::gpu::DrawCmd& cmd : t->frame.cmds) {  // the quads stay inside dest
            cmd.clipX = static_cast<int16_t>(std::min(cx0, 32767));
            cmd.clipY = static_cast<int16_t>(std::min(cy0, 32767));
            cmd.clipW = static_cast<int16_t>(std::min(cx1 - cx0, 32767));
            cmd.clipH = static_cast<int16_t>(std::min(cy1 - cy0, 32767));
        }
        const std::string error = submitComposite(*t);
        if (!error.empty()) return fail(XT_STATUS_INTERNAL, "xt_surface_draw_layers_gpu: " + error);
        return XT_STATUS_OK;
    });
}

XT_CAPI xt_status xt_surface_push_input_ray(xt_surface* s, const xt_input_event* e, const float origin[3],
                                            const float direction[3]) {
    XT_CAPI_GUARD("xt_surface_push_input_ray", {
        XT_CAPI_REQUIRE_SURFACE(s, "xt_surface_push_input_ray");
        if (e == nullptr || e->struct_size < sizeof(xt_input_event) || origin == nullptr ||
            direction == nullptr) {
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_surface_push_input_ray: event, origin or direction is NULL, or struct_size too small");
        }
        if (e->kind != XT_INPUT_POINTER_MOVE && e->kind != XT_INPUT_POINTER_DOWN &&
            e->kind != XT_INPUT_POINTER_UP && e->kind != XT_INPUT_WHEEL) {
            return fail(
                XT_STATUS_INVALID_ARGUMENT,
                "xt_surface_push_input_ray: kind must be POINTER_MOVE, POINTER_DOWN, POINTER_UP or WHEEL");
        }
        for (int i = 0; i < 3; ++i) {
            if (!std::isfinite(origin[i]) || !std::isfinite(direction[i])) {
                return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_push_input_ray: the ray is not finite");
            }
        }
        if (direction[0] == 0.0f && direction[1] == 0.0f && direction[2] == 0.0f) {
            return fail(XT_STATUS_INVALID_ARGUMENT, "xt_surface_push_input_ray: direction is zero");
        }
        if (!s->worldLayers) {
            return fail(XT_STATUS_INVALID_ARGUMENT,
                        "xt_surface_push_input_ray: world layers are off (xt_surface_set_world_layers)");
        }
        xt::InputEvent event;
        if (!xt::capi::convertInput(*e, event, "xt_surface_push_input_ray"))
            return XT_STATUS_INVALID_ARGUMENT;
        s->ctx->pushPointerRay(event, xt::Vec3{origin[0], origin[1], origin[2]},
                               xt::Vec3{direction[0], direction[1], direction[2]});
        ++s->debugInputCount;
        return XT_STATUS_OK;
    });
}

}  // extern "C"

#endif  // XT_CAPI_HAS_WGPU
