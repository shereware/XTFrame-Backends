#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "wgpu_internal.h"
#include "xt/wgpu/wgpu_device.h"

/// Command encoding of WgpuDevice::submit (T-438, Docs/design/10-backends.md §10.10.2): per draw
/// command the cached pipeline, the parameter and analytic clip offsets (group 1), the texture bind group
/// (group 2), the scissor (gl_device.cpp applyScissor in top-down pixels) and the stencil reference
/// (gl_device.cpp applyStencil's `ref`).

namespace xt::wgpu {

uint64_t detail::newViewId() {
    static std::atomic<uint64_t> next{kDummyViewId + 1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

namespace {

/// T-670: the non-sampler bindings of a cached group of `key` (a blit group: the uniform buffer and
/// one view; a texture group: three views).
uint32_t bindingsOf(const detail::BindGroupKey& key, WGPUBindGroupLayout blitLayout) {
    return key.layout == blitLayout ? 2u : 3u;
}

}  // namespace

void WgpuDevice::Gpu::eraseCached(
    std::unordered_map<detail::BindGroupKey, WGPUBindGroup, detail::BindGroupKeyHash>::iterator it,
    uint64_t skipId) {
    const detail::BindGroupKey key = it->first;
    wgpuBindGroupRelease(it->second);
    cachedBindings -= std::min(cachedBindings, bindingsOf(key, pipelines->blitLayout()));
    bindGroups.erase(it);
    const uint64_t ids[3] = {key.id0, key.id1, key.id2};
    for (int i = 0; i < 3; ++i) {
        const uint64_t id = ids[i];
        if (id == skipId || id == detail::kDummyViewId || (i > 0 && id == ids[0]) ||
            (i > 1 && id == ids[1])) {
            continue;
        }
        auto list = keysByView.find(id);
        if (list == keysByView.end()) continue;
        std::vector<detail::BindGroupKey>& keys = list->second;
        for (size_t k = 0; k < keys.size(); ++k) {
            if (keys[k] == key) {
                keys[k] = keys.back();
                keys.pop_back();
                break;
            }
        }
        if (keys.empty()) keysByView.erase(list);
    }
}

uint32_t WgpuDevice::Gpu::evictView(uint64_t id) {
    auto list = keysByView.find(id);
    if (list == keysByView.end()) {
        return 0;
    }
    const std::vector<detail::BindGroupKey> keys = std::move(list->second);
    keysByView.erase(list);
    uint32_t n = 0;
    for (const detail::BindGroupKey& key : keys) {
        auto it = bindGroups.find(key);
        if (it == bindGroups.end()) continue;
        eraseCached(it, id);
        ++n;
    }
    return n;
}

uint32_t WgpuDevice::Gpu::evictLayout(WGPUBindGroupLayout layout) {
    std::vector<detail::BindGroupKey> keys;
    for (const auto& kv : bindGroups) {
        if (kv.first.layout == layout) keys.push_back(kv.first);
    }
    for (const detail::BindGroupKey& key : keys) {
        eraseCached(bindGroups.find(key), 0);
    }
    return static_cast<uint32_t>(keys.size());
}

void WgpuDevice::forgetView(uint64_t id, bool shared) {
    if (id == 0) {
        return;
    }
    if (m_gpu) m_evictedBindGroups += m_gpu->evictView(id);
    if (shared && m_sharing != nullptr) {
        syncSharedViews();  // what the others logged before this one
        WgpuSharedCache::Impl& c = *m_sharing->m_impl;
        c.noteViewDestroyed(id);
        m_seenViewGeneration = c.viewGeneration;  // this device's own groups are gone already
    }
}

void WgpuDevice::syncSharedViews() {
    if (m_sharing == nullptr) {
        return;
    }
    WgpuSharedCache::Impl& c = *m_sharing->m_impl;
    if (m_seenViewGeneration == c.viewGeneration) {
        return;
    }
    const uint64_t first = c.viewGeneration - c.viewLog.size();  // the generation of viewLog[0]
    if (m_gpu) {
        if (m_seenViewGeneration < first) {
            m_evictedBindGroups += static_cast<uint32_t>(m_gpu->bindGroups.size());  // the log moved on
            m_gpu->releaseBindGroups();
        } else {
            for (size_t i = static_cast<size_t>(m_seenViewGeneration - first); i < c.viewLog.size(); ++i) {
                m_evictedBindGroups += m_gpu->evictView(c.viewLog[i]);
            }
        }
    }
    m_seenViewGeneration = c.viewGeneration;
}

WGPUBindGroup WgpuDevice::Gpu::bindGroup(WGPUDevice device, const detail::BindGroupKey& key,
                                         uint32_t& creates, bool perFrame) {
    // T-726: a key naming a view without an id (none should) is made for this submit only.
    if (key.id0 == 0 || (key.v1 != nullptr && key.id1 == 0) || (key.v2 != nullptr && key.id2 == 0)) {
        perFrame = true;
    }
    if (!perFrame) {
        auto it = bindGroups.find(key);
        if (it != bindGroups.end()) {
            return it->second;
        }
        if (bindGroups.size() >= kMaxBindGroups) {
            for (auto& kv : bindGroups) {
                wgpuBindGroupRelease(kv.second);
            }
            bindGroups.clear();
            keysByView.clear();  // T-726
            cachedBindings = 0;  // T-670
        }
    }
    WGPUBindGroupEntry e[4] = {WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT,
                               WGPU_BIND_GROUP_ENTRY_INIT, WGPU_BIND_GROUP_ENTRY_INIT};
    size_t count = 0;
    if (key.layout == pipelines->blitLayout()) {
        e[0].binding = 0;
        e[0].buffer = uniforms.buffer;
        e[0].size = 16;
        e[1].binding = 1;
        e[1].textureView = key.v0;
        count = 2;
    } else {
        e[0].binding = 0;
        e[0].textureView = key.v0;
        e[1].binding = 1;
        e[1].textureView = key.v1;
        e[2].binding = 2;
        e[2].textureView = key.v2;
        e[3].binding = 3;
        e[3].sampler = key.sampler;
        count = 4;
    }
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = key.layout;
    d.entryCount = count;
    d.entries = e;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(device, &d);
    ++creates;
    for (size_t i = 0; i < count; ++i) {  // T-670: live non-sampler bindings
        (perFrame ? frameBindings : cachedBindings) += e[i].sampler == nullptr ? 1u : 0u;
    }
    noteBindings();
    if (perFrame) {
        frameBindGroups.push_back(bg);
    } else {
        bindGroups.emplace(key, bg);
        const uint64_t ids[3] = {key.id0, key.id1, key.id2};  // T-726: indexed by each distinct id
        for (int i = 0; i < 3; ++i) {
            const uint64_t id = ids[i];
            if (id == 0 || id == detail::kDummyViewId || (i > 0 && id == ids[0]) || (i > 1 && id == ids[1])) {
                continue;
            }
            keysByView[id].push_back(key);
        }
    }
    return bg;
}

void WgpuDevice::planClipBlocks(const gpu::GpuFrame& frame) {
    m_cmdClipOffsets.assign(frame.cmds.size(), 0);
    const size_t clipFloats = detail::kClipBlockBytes / sizeof(float);
    const size_t slotFloats = detail::kUniformAlign / sizeof(float);  // one 256-byte step holds a block
    static_assert(detail::kClipBlockBytes <= detail::kUniformAlign, "a clip block is one aligned step");
    uint32_t lastIndex = 0, lastOffset = 0;
    std::unordered_map<uint32_t, uint32_t> byIndex;
    for (size_t c = 0; c < frame.cmds.size(); ++c) {
        const uint32_t index = frame.cmds[c].clipParamIndex;
        if (index == 0) {
            continue;  // offset 0: the zero block (count 0)
        }
        if (index == lastIndex) {
            m_cmdClipOffsets[c] = lastOffset;
            continue;
        }
        auto it = byIndex.find(index);
        if (it == byIndex.end()) {
            if (static_cast<size_t>(index) + clipFloats > frame.params.size()) {
                m_cmdParamOffsets[c] = kNoParams;  // a block outside the pool: the command is not drawn
                continue;
            }
            const uint32_t offset = static_cast<uint32_t>(m_uniformData.size() * sizeof(float));
            m_uniformData.resize(m_uniformData.size() + slotFloats, 0.0f);
            std::memcpy(&m_uniformData[offset / 4], frame.params.data() + index, detail::kClipBlockBytes);
            it = byIndex.emplace(index, offset).first;
        }
        lastIndex = index;
        lastOffset = it->second;
        m_cmdClipOffsets[c] = lastOffset;
    }
}

void WgpuDevice::blit(WGPURenderPassEncoder pass, WGPUTextureView src, uint64_t srcId, uint32_t uniformOffset,
                      int x, int y, int w, int h, WGPUTextureFormat format, uint32_t samples, bool hasStencil,
                      uint32_t targetW, uint32_t targetH) {
    Gpu& g = *m_gpu;
    WGPURenderPipeline p = g.pipelines->blit(format, samples, hasStencil);
    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, static_cast<int>(targetW) - x);
    h = std::min(h, static_cast<int>(targetH) - y);
    if (p == nullptr || src == nullptr || w <= 0 || h <= 0) {
        ++m_stats.skippedCopies;
        return;
    }
    detail::BindGroupKey key;
    key.layout = g.pipelines->blitLayout();
    key.v0 = src;
    key.id0 = src == m_surface ? 0 : srcId;  // T-726: the surface view's group is this submit's only
    WGPUBindGroup bg = g.bindGroup(m_device, key, m_stats.bindGroups, src == m_surface);
    wgpuRenderPassEncoderSetPipeline(pass, p);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bg, 1, &uniformOffset);
    wgpuRenderPassEncoderSetViewport(pass, 0.0f, 0.0f, static_cast<float>(targetW),
                                     static_cast<float>(targetH), 0.0f, 1.0f);
    wgpuRenderPassEncoderSetScissorRect(pass, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                                        static_cast<uint32_t>(w), static_cast<uint32_t>(h));
    wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
    ++m_stats.drawCalls;
    ++m_stats.pipelineBinds;
}

void WgpuDevice::drawCmd(WGPURenderPassEncoder pass, const gpu::DrawCmd& cmd, uint32_t cmdIndex,
                         const PassPlan& plan, gpu::TargetHandle target) {
    using gpu::Shading;
    using Clock = std::chrono::steady_clock;
    Gpu& g = *m_gpu;
    const gpu::GpuFrame& frame = *m_frame;
    const bool particles = cmd.kind == gpu::DrawKind::Particles;
    if (particles ? (cmd.instanceCount == 0 || static_cast<size_t>(cmd.firstInstance) + cmd.instanceCount >
                                                   frame.particleInstances.size())
                  : cmd.indexCount == 0) {
        return;
    }
    const bool timed = cmd.shading == Shading::Effect || cmd.shading == Shading::Glyph;
    const Clock::time_point t0 = timed ? Clock::now() : Clock::time_point{};
    const uint32_t paramOffset = m_cmdParamOffsets[cmdIndex];
    if (paramOffset == kNoParams) {
        ++m_stats.skippedCmds;
        return;
    }

    // Target attachments of the open pass.
    WGPUTextureFormat format = m_targetFormat;
    uint32_t samples = m_sampleCount;
    bool hasStencil = surfaceStencilView() != nullptr;  // T-703: a layer target may have none
    if (target != gpu::kSurfaceTarget) {
        const Pooled& p = pool()[static_cast<size_t>(m_frameTargets[target - 1])];
        format = detail::formatOf(p.format);
        samples = p.samples;
        hasStencil = p.depthStencilView != nullptr;
    }

    // The fragment program.
    uint16_t shader = detail::kShSolid;
    switch (cmd.shading) {
        case Shading::Solid:
            shader = cmd.coverage != 0 ? detail::kShCoverageSolid : detail::kShSolid;  // T-660b
            break;
        case Shading::StencilOnly:
            shader = detail::kShSolid;
            break;
        case Shading::Gradient:
            shader = cmd.coverage != 0 ? detail::kShCoverageGradient : detail::kShGradient;
            break;
        case Shading::Texture:  // T-665: a mip draw samples at its explicit lod (TextureLodParams)
            shader = cmd.mipLod != 0 ? detail::kShTextureLod : detail::kShTexture;
            break;
        case Shading::TextureStraight:
            shader = cmd.mipLod != 0 ? detail::kShTextureLod : detail::kShTextureStraight;
            break;
        case Shading::Glyph:
            shader = detail::kShGlyph;
            break;
        case Shading::Composite:
            shader = detail::kShComposite;
            break;
        case Shading::ParticleTexture:
            shader = detail::kShParticleTexture;
            break;
        case Shading::ParticleSolid:
            shader = detail::kShParticleSolid;
            break;
        case Shading::BoxShadow:
            // T-668; T-683: the builder chose the one-shape or the up-to-three-shape program.
            shader = cmd.boxShadowMulti != 0 ? detail::kShBoxShadowMulti : detail::kShBoxShadow;
            break;
        case Shading::Effect:
            if (cmd.program == gpu::EffectProgram::Custom) {
                const float* p = &frame.params[cmd.paramIndex];  // packCustom checked the block
                const uint32_t id = static_cast<uint32_t>(p[0]);
                if (id == 0 || id > g.pipelines->customCount()) {
                    m_lastError = "unknown custom effect program";
                    ++m_stats.skippedCmds;
                    return;
                }
                shader = static_cast<uint16_t>(detail::kShaderCount + id - 1);
            } else {
                shader = detail::effectShader(cmd.program);
                if (shader == detail::kShaderCount) {
                    m_lastError = "unsupported effect program";
                    ++m_stats.skippedCmds;
                    return;
                }
            }
            break;
    }
    if (particles != (cmd.shading == Shading::ParticleTexture || cmd.shading == Shading::ParticleSolid)) {
        return;  // gl_device.cpp: particle shadings draw only as DrawKind::Particles and vice versa
    }

    detail::PipelineKey key;
    key.shader = shader;
    key.blend = cmd.blend;
    key.stencil = hasStencil ? cmd.stencilOp : gpu::StencilOp::None;
    key.kind = cmd.kind;
    key.colorWrite = cmd.shading != Shading::StencilOnly;
    key.format = format;
    key.samples = static_cast<uint8_t>(samples);
    key.hasStencil = hasStencil;
    WGPURenderPipeline pipeline = g.pipelines->get(key);
    if (pipeline == nullptr) {
        ++m_stats.skippedCmds;
        return;
    }

    // Scissor: surface pixels clipped to the pass window, then into the target's own pixels.
    const int x0 = std::max<int>(cmd.clipX, plan.win.x);
    const int y0 = std::max<int>(cmd.clipY, plan.win.y);
    const int x1 = std::min<int>(cmd.clipX + cmd.clipW, plan.win.x + plan.win.w);
    const int y1 = std::min<int>(cmd.clipY + cmd.clipH, plan.win.y + plan.win.h);
    if (x1 <= x0 || y1 <= y0) {
        return;
    }
    int sx0, sy0, sx1, sy1;
    if (plan.scaleX != 1.0f || plan.scaleY != 1.0f) {
        // T-343: a scaled transient; the window maps onto the whole (smaller) target.
        sx0 = static_cast<int>(std::floor(static_cast<float>(x0 - plan.originX) * plan.scaleX));
        sx1 = static_cast<int>(std::ceil(static_cast<float>(x1 - plan.originX) * plan.scaleX));
        sy0 = static_cast<int>(std::floor(static_cast<float>(y0 - plan.originY) * plan.scaleY));
        sy1 = static_cast<int>(std::ceil(static_cast<float>(y1 - plan.originY) * plan.scaleY));
    } else {
        sx0 = x0 - plan.originX;
        sx1 = x1 - plan.originX;
        sy0 = y0 - plan.originY;
        sy1 = y1 - plan.originY;
    }
    sx0 = std::max(sx0, 0);
    sy0 = std::max(sy0, 0);
    sx1 = std::min(sx1, plan.targetW);
    sy1 = std::min(sy1, plan.targetH);
    if (sx1 <= sx0 || sy1 <= sy0) {
        return;
    }

    // Group 2: the textures and sampler of this shading.
    detail::BindGroupKey bk;
    bk.layout = g.pipelines->textureLayout(shader);
    bk.v0 = bk.v1 = bk.v2 = g.dummyView;
    bk.id0 = bk.id1 = bk.id2 = detail::kDummyViewId;  // T-726
    bk.sampler = g.pipelines->sampler(gpu::Sampler::LinearClamp);
    // T-726: each slot gets the view and its id.
    auto textureView = [this, &g](gpu::TextureHandle h, WGPUTextureView& v, uint64_t& id) {
        const Texture* t = find(h);
        if (t == nullptr) {
            v = g.dummyView;
            id = detail::kDummyViewId;
            return;
        }
        v = viewOf(*t);  // T-665: a shared image's current view
        id = viewIdOf(*t);
    };
    auto targetView = [this, &g](gpu::TargetHandle t, WGPUTextureView& v, uint64_t& id) {
        v = sampleView(t);
        id = sampleViewId(t);
        if (t == gpu::kSurfaceTarget || v == nullptr) {
            v = g.dummyView;
            id = detail::kDummyViewId;
        }
    };
    switch (cmd.shading) {
        case Shading::Texture:
        case Shading::TextureStraight:
        case Shading::ParticleTexture:
            textureView(cmd.texture, bk.v0, bk.id0);
            bk.sampler = g.pipelines->sampler(cmd.sampler);
            break;
        case Shading::Gradient:
        case Shading::Glyph:
            textureView(cmd.texture, bk.v0, bk.id0);  // gl_device.cpp: always LinearClamp
            if (cmd.shading == Shading::Gradient && cmd.coverage != 0) {
                textureView(cmd.coverage, bk.v1, bk.id1);  // T-660b: t1 = the coverage page
            }
            break;
        case Shading::Solid:
            if (cmd.coverage != 0) {
                textureView(cmd.coverage, bk.v0, bk.id0);  // T-660b: t0 = the coverage page, LinearClamp
            }
            break;
        case Shading::Composite:
        case Shading::Effect:
            targetView(cmd.src, bk.v0, bk.id0);
            if (cmd.aux != 0) targetView(cmd.aux, bk.v1, bk.id1);
            if (cmd.texture != 0) textureView(cmd.texture, bk.v2, bk.id2);
            if (cmd.program == gpu::EffectProgram::BackdropMasked) {  // T-729: t2 = the content (S)
                targetView(gpu::backdropMaskTarget(frame, cmd), bk.v2, bk.id2);
            }
            break;
        default:
            break;
    }
    WGPUBindGroup textures = g.bindGroup(m_device, bk, m_stats.bindGroups);

    if (pipeline != g.pipeline) {
        wgpuRenderPassEncoderSetPipeline(pass, pipeline);
        g.pipeline = pipeline;
        ++m_stats.pipelineBinds;
    }
    const uint32_t clipOffset = m_cmdClipOffsets[cmdIndex];  // T-718: the zero block without a clip
    if (paramOffset != g.paramOffset || clipOffset != g.clipOffset) {
        const uint32_t offsets[2] = {paramOffset, clipOffset};
        wgpuRenderPassEncoderSetBindGroup(pass, 1, g.paramGroup, 2, offsets);
        g.paramOffset = paramOffset;
        g.clipOffset = clipOffset;
    }
    if (textures != g.textures) {
        wgpuRenderPassEncoderSetBindGroup(pass, 2, textures, 0, nullptr);
        g.textures = textures;
    }
    if (hasStencil) {
        // gl_device.cpp applyStencil: the depth ops compare against `ref << 4`; the cover test and
        // the winding clear use 0.
        const bool zeroRef =
            cmd.stencilOp == gpu::StencilOp::TestNotEqualZero || cmd.stencilOp == gpu::StencilOp::ClearRect;
        const uint32_t ref = zeroRef ? 0u : (static_cast<uint32_t>(cmd.stencilRef) << 4) & 0xFFu;
        if (ref != g.stencilRef) {
            wgpuRenderPassEncoderSetStencilReference(pass, ref);
            g.stencilRef = ref;
        }
    }
    wgpuRenderPassEncoderSetScissorRect(pass, static_cast<uint32_t>(sx0), static_cast<uint32_t>(sy0),
                                        static_cast<uint32_t>(sx1 - sx0), static_cast<uint32_t>(sy1 - sy0));
    if (particles) {
        if (g.vertexSource != 2) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.geometry.instances(), 0,
                                                 g.geometry.instanceBytes());
            g.vertexSource = 2;
        }
        wgpuRenderPassEncoderDraw(pass, 6, cmd.instanceCount, 0, cmd.firstInstance);
    } else {
        if (g.vertexSource != 1) {
            wgpuRenderPassEncoderSetVertexBuffer(pass, 0, g.geometry.vertices(), 0, g.geometry.vertexBytes());
            g.vertexSource = 1;
        }
        wgpuRenderPassEncoderDrawIndexed(pass, cmd.indexCount, 1, cmd.firstIndex,
                                         static_cast<int32_t>(cmd.baseVertex), 0);
    }
    ++m_stats.drawCalls;
    if (timed) {
        const float ms = std::chrono::duration<float, std::milli>(Clock::now() - t0).count();
        (cmd.shading == Shading::Effect ? m_stats.effectsMs : m_stats.textMs) += ms;
    }
}

}  // namespace xt::wgpu
