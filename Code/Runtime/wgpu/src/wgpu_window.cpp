#include "xt/wgpu/wgpu_window.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <objc/message.h>
#include <objc/runtime.h>
#endif

/// WgpuWindowSurface (T-636): the native window -> WGPUSurface code of Code/Backends' RendererWgpu
/// (T-439), moved here so the C ABI GPU target (50 §50.z) uses the same one.

namespace xt::wgpu {

namespace {

bool hasFormat(const WGPUSurfaceCapabilities& caps, WGPUTextureFormat format) {
    for (size_t i = 0; i < caps.formatCount; ++i) {
        if (caps.formats[i] == format) return true;
    }
    return false;
}

bool hasPresentMode(const WGPUSurfaceCapabilities& caps, WGPUPresentMode mode) {
    for (size_t i = 0; i < caps.presentModeCount; ++i) {
        if (caps.presentModes[i] == mode) return true;
    }
    return false;
}

#if defined(__APPLE__)
/// A new CAMetalLayer set as the layer of `nsWindow`'s content view (a layer-hosting view), through
/// the Objective-C runtime so this file stays C++. Null when the window has no content view or
/// QuartzCore is not loaded. Untested (no macOS machine, T-636).
void* metalLayerOfWindow(void* nsWindow) {
    using SendId = void* (*)(void*, SEL);
    using SendSetId = void (*)(void*, SEL, void*);
    using SendSetBool = void (*)(void*, SEL, signed char);
    const auto sendId = reinterpret_cast<SendId>(objc_msgSend);
    void* view = sendId(nsWindow, sel_registerName("contentView"));
    Class layerClass = objc_getClass("CAMetalLayer");
    if (view == nullptr || layerClass == nullptr) return nullptr;
    void* layer = sendId(reinterpret_cast<void*>(layerClass), sel_registerName("layer"));
    if (layer == nullptr) return nullptr;
    reinterpret_cast<SendSetId>(objc_msgSend)(view, sel_registerName("setLayer:"), layer);
    reinterpret_cast<SendSetBool>(objc_msgSend)(view, sel_registerName("setWantsLayer:"), 1);
    return layer;
}
#endif

/// T-758: the full-screen pass that moves the intermediate onto a surface offering neither
/// TEXTURE_BINDING nor COPY_DST -- one triangle covering the target, each pixel loaded unfiltered from
/// the same coordinate (same size and format: the bytes are kept).
constexpr char kSurfaceBlitWgsl[] = R"(
@group(0) @binding(0) var src: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
@fragment fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(src, vec2<i32>(pos.xy), 0);
}
)";

/// "RenderAttachment|CopySrc|..." for the notes.
std::string usageText(WGPUTextureUsage usage) {
    struct Bit {
        WGPUTextureUsage bit;
        const char* name;
    };
    const Bit bits[] = {{WGPUTextureUsage_RenderAttachment, "RenderAttachment"},
                        {WGPUTextureUsage_CopySrc, "CopySrc"},
                        {WGPUTextureUsage_CopyDst, "CopyDst"},
                        {WGPUTextureUsage_TextureBinding, "TextureBinding"},
                        {WGPUTextureUsage_StorageBinding, "StorageBinding"},
                        {WGPUTextureUsage_TransientAttachment, "TransientAttachment"}};
    std::string out;
    for (const Bit& b : bits) {
        if ((usage & b.bit) != 0) out += (out.empty() ? "" : "|") + std::string(b.name);
    }
    return out.empty() ? "None" : out;
}

}  // namespace

SurfaceUsageChoice chooseSurfaceUsage(WGPUTextureUsage capsUsages, SurfaceReads reads) {
    SurfaceUsageChoice c;
    c.usage = WGPUTextureUsage_RenderAttachment | (capsUsages & WGPUTextureUsage_CopySrc) |
              (capsUsages & WGPUTextureUsage_TextureBinding);
    c.readable = (capsUsages & WGPUTextureUsage_TextureBinding) != 0;
    c.direct = c.readable || reads == SurfaceReads::None;  // T-749
    if (!c.direct) c.usage |= capsUsages & WGPUTextureUsage_CopyDst;
    return c;
}

WGPUTexture createRenderTexture(WGPUDevice device, int width, int height, WGPUTextureFormat format,
                                const char* label) {
    if (device == nullptr || width <= 0 || height <= 0) return nullptr;
    WGPUTextureDescriptor d = WGPU_TEXTURE_DESCRIPTOR_INIT;
    if (label != nullptr) d.label = WGPUStringView{label, WGPU_STRLEN};
    d.usage = WGPUTextureUsage_RenderAttachment | WGPUTextureUsage_CopySrc | WGPUTextureUsage_TextureBinding;
    d.size = WGPUExtent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    d.format = format;
    return wgpuDeviceCreateTexture(device, &d);
}

WgpuWindowSurface::~WgpuWindowSurface() { release(); }

bool WgpuWindowSurface::create(const WgpuInit& init, const WgpuNativeWindow& window, bool vsync,
                               std::string& error, SurfaceReads reads) {
    release();
    if (init.instance == nullptr || init.adapter == nullptr) {
        error = "no WebGPU instance / adapter for the window surface";
        return false;
    }
    WGPUSurfaceDescriptor desc = WGPU_SURFACE_DESCRIPTOR_INIT;
    WGPUSurfaceSourceWindowsHWND win32 = WGPU_SURFACE_SOURCE_WINDOWS_HWND_INIT;
    WGPUSurfaceSourceXlibWindow xlib = WGPU_SURFACE_SOURCE_XLIB_WINDOW_INIT;
    WGPUSurfaceSourceWaylandSurface wayland = WGPU_SURFACE_SOURCE_WAYLAND_SURFACE_INIT;
    WGPUSurfaceSourceMetalLayer metal = WGPU_SURFACE_SOURCE_METAL_LAYER_INIT;
    switch (window.kind) {
        case WgpuNativeWindow::Kind::Win32:
            win32.hinstance = window.display;
#if defined(_WIN32)
            if (win32.hinstance == nullptr) win32.hinstance = GetModuleHandleW(nullptr);
#endif
            win32.hwnd = window.window;
            desc.nextInChain = &win32.chain;
            break;
        case WgpuNativeWindow::Kind::X11:
            xlib.display = window.display;
            xlib.window = window.x11Window;
            desc.nextInChain = &xlib.chain;
            break;
        case WgpuNativeWindow::Kind::Wayland:
            wayland.display = window.display;
            wayland.surface = window.window;
            desc.nextInChain = &wayland.chain;
            break;
        case WgpuNativeWindow::Kind::MetalLayer:
            metal.layer = window.window;
            desc.nextInChain = &metal.chain;
            break;
        case WgpuNativeWindow::Kind::CocoaWindow:
#if defined(__APPLE__)
            metal.layer = window.window != nullptr ? metalLayerOfWindow(window.window) : nullptr;
            if (metal.layer == nullptr) {
                error = "the NSWindow gives no CAMetalLayer (no content view?)";
                return false;
            }
            desc.nextInChain = &metal.chain;
            break;
#else
            error = "a Cocoa window on a platform without Cocoa";
            return false;
#endif
        case WgpuNativeWindow::Kind::None:
            error = "the window has no native handle for a WebGPU surface";
            return false;
    }
    if (window.kind != WgpuNativeWindow::Kind::X11 && window.window == nullptr) {
        error = "the native window handle is null";
        return false;
    }
    m_surface = wgpuInstanceCreateSurface(init.instance, &desc);
    if (m_surface == nullptr) {
        error = "wgpuInstanceCreateSurface failed";
        return false;
    }
    WGPUSurfaceCapabilities caps = WGPU_SURFACE_CAPABILITIES_INIT;
    if (wgpuSurfaceGetCapabilities(m_surface, init.adapter, &caps) != WGPUStatus_Success ||
        caps.formatCount == 0) {
        wgpuSurfaceCapabilitiesFreeMembers(caps);
        error = "the adapter cannot present to this window";
        release();
        return false;
    }
    m_format = hasFormat(caps, WGPUTextureFormat_BGRA8Unorm)   ? WGPUTextureFormat_BGRA8Unorm
               : hasFormat(caps, WGPUTextureFormat_RGBA8Unorm) ? WGPUTextureFormat_RGBA8Unorm
                                                               : caps.formats[0];
    m_offered = caps.usages;
    const SurfaceUsageChoice choice = chooseSurfaceUsage(caps.usages, reads);  // T-758, T-749
    m_usage = choice.usage;
    m_readable = choice.readable;
    m_direct = choice.direct;
    m_presentMode = WGPUPresentMode_Fifo;
    if (!vsync) {
        if (hasPresentMode(caps, WGPUPresentMode_Immediate)) {
            m_presentMode = WGPUPresentMode_Immediate;
        } else if (hasPresentMode(caps, WGPUPresentMode_Mailbox)) {
            m_presentMode = WGPUPresentMode_Mailbox;
        }
    }
    wgpuSurfaceCapabilitiesFreeMembers(caps);
    return true;
}

bool WgpuWindowSurface::createForTesting(WGPUDevice device, WGPUTexture texture, std::string& error,
                                         SurfaceReads reads) {
    if (device == nullptr || texture == nullptr) {
        error = "createForTesting: no device or texture";
        return false;
    }
    if (m_testTexture == nullptr || m_device != device || wgpuTextureGetFormat(texture) != m_format) {
        release();
    } else {
        releaseFrame();
    }
    m_testTexture = texture;
    m_device = device;
    m_format = wgpuTextureGetFormat(texture);
    m_offered = wgpuTextureGetUsage(texture);
    const SurfaceUsageChoice choice = chooseSurfaceUsage(m_offered, reads);
    m_usage = choice.usage;
    m_readable = choice.readable;
    m_direct = choice.direct;
    m_width = static_cast<int>(wgpuTextureGetWidth(texture));
    m_height = static_cast<int>(wgpuTextureGetHeight(texture));
    return true;
}

void WgpuWindowSurface::configure(WGPUDevice device, int width, int height) {
    if (device == nullptr) return;
    if (m_testTexture != nullptr) {  // T-758 tests: the stand-in keeps its size
        m_device = device;
        return;
    }
    if (m_surface == nullptr) return;
    m_device = device;
    m_width = width > 0 ? width : 1;
    m_height = height > 0 ? height : 1;
    WGPUSurfaceConfiguration config = WGPU_SURFACE_CONFIGURATION_INIT;
    config.device = device;
    config.format = m_format;
    config.usage = m_usage;
    config.width = static_cast<uint32_t>(m_width);
    config.height = static_cast<uint32_t>(m_height);
    config.alphaMode = WGPUCompositeAlphaMode_Auto;
    config.presentMode = m_presentMode;
    wgpuSurfaceConfigure(m_surface, &config);
}

bool WgpuWindowSurface::acquire() {
    releaseFrame();
    if (m_device == nullptr) return false;
    if (m_testTexture != nullptr) {
        wgpuTextureAddRef(m_testTexture);
        m_surfaceTexture = m_testTexture;
    } else {
        if (m_surface == nullptr) return false;
        for (int attempt = 0; attempt < 2 && m_surfaceTexture == nullptr; ++attempt) {
            WGPUSurfaceTexture st = WGPU_SURFACE_TEXTURE_INIT;
            wgpuSurfaceGetCurrentTexture(m_surface, &st);
            if (st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessOptimal ||
                st.status == WGPUSurfaceGetCurrentTextureStatus_SuccessSuboptimal) {
                m_surfaceTexture = st.texture;
                break;
            }
            if (st.texture != nullptr) wgpuTextureRelease(st.texture);
            if (st.status != WGPUSurfaceGetCurrentTextureStatus_Outdated &&
                st.status != WGPUSurfaceGetCurrentTextureStatus_Lost) {
                return false;  // timeout / error: skip this frame
            }
            configure(m_device, m_width, m_height);
        }
        if (m_surfaceTexture == nullptr) return false;
    }
    if (m_direct) {  // T-749: readable, or frames that never sample it
        m_surfaceView = wgpuTextureCreateView(m_surfaceTexture, nullptr);
        m_frameTexture = m_surfaceTexture;
        m_frameView = m_surfaceView;
        return true;
    }
    // T-758: not readable -- the frame is drawn into the intermediate, of the texture's actual size.
    if (!ensureIntermediate(wgpuTextureGetWidth(m_surfaceTexture), wgpuTextureGetHeight(m_surfaceTexture))) {
        releaseFrame();
        return false;
    }
    m_frameTexture = m_intermediate;
    m_frameView = m_intermediateView;
    return true;
}

void WgpuWindowSurface::present() {
    if (m_surfaceTexture != nullptr) {
        if (!m_direct) transferToSurface();
        if (m_surface != nullptr) wgpuSurfacePresent(m_surface);
    }
    releaseFrame();
}

bool WgpuWindowSurface::frameCopyable() const {
    return !m_direct || (m_usage & WGPUTextureUsage_CopySrc) != 0;
}

std::string WgpuWindowSurface::usageNote() const {
    std::string how = "frames drawn into the surface";
    if (m_direct && !m_readable) {
        how += " (they never sample it: no intermediate texture)";  // T-749
    } else if (!m_direct) {
        how = std::string("no TextureBinding: frames drawn into an intermediate texture, ") +
              ((m_usage & WGPUTextureUsage_CopyDst) != 0 ? "copied" : "drawn by a full-screen pass") +
              " to the surface at present";
    }
    return "surface usages offered " + usageText(m_offered) + ", configured " + usageText(m_usage) + "; " +
           how;
}

void WgpuWindowSurface::releaseFrame() {
    if (m_surfaceView != nullptr) wgpuTextureViewRelease(m_surfaceView);
    if (m_surfaceTexture != nullptr) wgpuTextureRelease(m_surfaceTexture);
    m_surfaceView = nullptr;
    m_surfaceTexture = nullptr;
    m_frameView = nullptr;
    m_frameTexture = nullptr;
}

bool WgpuWindowSurface::ensureIntermediate(uint32_t width, uint32_t height) {
    if (m_intermediate != nullptr && wgpuTextureGetWidth(m_intermediate) == width &&
        wgpuTextureGetHeight(m_intermediate) == height) {
        return true;
    }
    releaseIntermediate();
    m_intermediate = createRenderTexture(m_device, static_cast<int>(width), static_cast<int>(height),
                                         m_format, "xt_wgpu window intermediate");
    if (m_intermediate == nullptr) return false;
    m_intermediateView = wgpuTextureCreateView(m_intermediate, nullptr);
    return m_intermediateView != nullptr;
}

void WgpuWindowSurface::releaseIntermediate() {
    if (m_intermediateView != nullptr) wgpuTextureViewRelease(m_intermediateView);
    if (m_intermediate != nullptr) {
        wgpuTextureDestroy(m_intermediate);
        wgpuTextureRelease(m_intermediate);
    }
    m_intermediateView = nullptr;
    m_intermediate = nullptr;
}

bool WgpuWindowSurface::ensureBlit() {
    if (m_blitPipeline != nullptr) return true;
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = WGPUStringView{kSurfaceBlitWgsl, WGPU_STRLEN};
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(m_device, &md);
    if (module == nullptr) return false;
    WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entry.binding = 0;
    entry.visibility = WGPUShaderStage_Fragment;
    entry.texture.sampleType = WGPUTextureSampleType_UnfilterableFloat;
    entry.texture.viewDimension = WGPUTextureViewDimension_2D;
    WGPUBindGroupLayoutDescriptor bd = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bd.entryCount = 1;
    bd.entries = &entry;
    m_blitLayout = wgpuDeviceCreateBindGroupLayout(m_device, &bd);
    WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pd.bindGroupLayoutCount = 1;
    pd.bindGroupLayouts = &m_blitLayout;
    WGPUPipelineLayout layout = wgpuDeviceCreatePipelineLayout(m_device, &pd);
    WGPUColorTargetState target = WGPU_COLOR_TARGET_STATE_INIT;
    target.format = m_format;
    target.writeMask = WGPUColorWriteMask_All;
    WGPUFragmentState fs = WGPU_FRAGMENT_STATE_INIT;
    fs.module = module;
    fs.entryPoint = WGPUStringView{"fs", WGPU_STRLEN};
    fs.targetCount = 1;
    fs.targets = &target;
    WGPURenderPipelineDescriptor d = WGPU_RENDER_PIPELINE_DESCRIPTOR_INIT;
    d.layout = layout;
    d.vertex.module = module;
    d.vertex.entryPoint = WGPUStringView{"vs", WGPU_STRLEN};
    d.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    d.primitive.cullMode = WGPUCullMode_None;
    d.multisample.count = 1;
    d.multisample.mask = 0xFFFFFFFFu;
    d.fragment = &fs;
    m_blitPipeline = wgpuDeviceCreateRenderPipeline(m_device, &d);
    wgpuPipelineLayoutRelease(layout);
    wgpuShaderModuleRelease(module);
    return m_blitPipeline != nullptr;
}

void WgpuWindowSurface::releaseBlit() {
    if (m_blitPipeline != nullptr) wgpuRenderPipelineRelease(m_blitPipeline);
    if (m_blitLayout != nullptr) wgpuBindGroupLayoutRelease(m_blitLayout);
    m_blitPipeline = nullptr;
    m_blitLayout = nullptr;
}

void WgpuWindowSurface::transferToSurface() {
    if (m_intermediate == nullptr || m_surfaceTexture == nullptr || m_device == nullptr) return;
    const bool copy = (m_usage & WGPUTextureUsage_CopyDst) != 0;
    if (!copy && !ensureBlit()) return;  // the frame cannot reach the surface
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(m_device, nullptr);
    WGPUBindGroup group = nullptr;
    if (copy) {
        WGPUTexelCopyTextureInfo s = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        s.texture = m_intermediate;
        WGPUTexelCopyTextureInfo d = WGPU_TEXEL_COPY_TEXTURE_INFO_INIT;
        d.texture = m_surfaceTexture;
        const WGPUExtent3D size{wgpuTextureGetWidth(m_intermediate), wgpuTextureGetHeight(m_intermediate), 1};
        wgpuCommandEncoderCopyTextureToTexture(enc, &s, &d, &size);
    } else {
        if (m_surfaceView == nullptr) m_surfaceView = wgpuTextureCreateView(m_surfaceTexture, nullptr);
        WGPUBindGroupEntry be = WGPU_BIND_GROUP_ENTRY_INIT;
        be.binding = 0;
        be.textureView = m_intermediateView;
        WGPUBindGroupDescriptor bgd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bgd.layout = m_blitLayout;
        bgd.entryCount = 1;
        bgd.entries = &be;
        group = wgpuDeviceCreateBindGroup(m_device, &bgd);
        WGPURenderPassColorAttachment ca = WGPU_RENDER_PASS_COLOR_ATTACHMENT_INIT;
        ca.view = m_surfaceView;
        ca.loadOp = WGPULoadOp_Clear;  // every pixel is written
        ca.storeOp = WGPUStoreOp_Store;
        ca.clearValue = WGPUColor{0, 0, 0, 0};
        WGPURenderPassDescriptor rp = WGPU_RENDER_PASS_DESCRIPTOR_INIT;
        rp.colorAttachmentCount = 1;
        rp.colorAttachments = &ca;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(enc, &rp);
        wgpuRenderPassEncoderSetPipeline(pass, m_blitPipeline);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
        wgpuRenderPassEncoderDraw(pass, 3, 1, 0, 0);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    WGPUQueue queue = wgpuDeviceGetQueue(m_device);
    wgpuQueueSubmit(queue, 1, &cmd);
    wgpuQueueRelease(queue);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    if (group != nullptr) wgpuBindGroupRelease(group);
}

void WgpuWindowSurface::release() {
    releaseFrame();
    releaseIntermediate();
    releaseBlit();
    if (m_surface != nullptr) {
        if (m_device != nullptr) wgpuSurfaceUnconfigure(m_surface);
        wgpuSurfaceRelease(m_surface);
        m_surface = nullptr;
    }
    m_testTexture = nullptr;
    m_device = nullptr;
}

}  // namespace xt::wgpu
