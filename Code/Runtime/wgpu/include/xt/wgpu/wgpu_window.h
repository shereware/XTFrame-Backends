#pragma once

#include <webgpu/webgpu.h>

#include <cstdint>
#include <string>

#include "xt/wgpu/wgpu_renderer.h"

/// A `WGPUSurface` made from a native window and presented to (Docs/design/10-backends.md §10.10,
/// T-439; shared by Code/Backends' RendererWgpu and the C ABI GPU target of 50 §50.z, T-636).
///
/// Format: the preferred non-sRGB 8-bit format (BGRA8Unorm, else RGBA8Unorm, else the first one
/// offered) -- the pipelines write premultiplied sRGB-encoded values as they are, like GL. Usage:
/// chooseSurfaceUsage. T-758 (10 §10.20 2026-11-18 note (1)): a frame samples target 0 (a mix-blend
/// backdrop copy, a backdrop blur pyramid, a Direct-mode scene read, a 4x surface loaded without a
/// clear), so the surface is drawn into directly only when it offers TEXTURE_BINDING ("readable"); on
/// one that does not, every frame is drawn into an intermediate texture (createRenderTexture, the
/// surface's size and format, made again when the surface's size changes) that present() copies to
/// the surface (when it offers COPY_DST) or draws onto it with a full-screen pass; frameTexture() /
/// frameView() are then the intermediate's, so a frame is read back from it.
/// T-749 (10 §10.10.2, the T-758 supplement's ruling): a user whose frames never sample target 0 (the C
/// ABI GPU target, whose composer only draws textured quads over it) creates the surface with
/// SurfaceReads::None: frames are then always drawn into the surface itself, and no intermediate is made.
/// Present mode: Fifo with vsync, else Immediate, else Mailbox, else Fifo. A texture that comes back
/// outdated or lost is reconfigured once and asked for again.

namespace xt::wgpu {

/// The usage a surface is configured with, from the usages it offers (`WGPUSurfaceCapabilities::usages`):
/// RENDER_ATTACHMENT; COPY_SRC when offered (a frame can be read back before present); TEXTURE_BINDING
/// when offered (`readable`: frames may sample it); when not readable, COPY_DST when offered (the
/// intermediate is copied in rather than drawn).
/// T-749: whether the frames drawn through the surface may sample it (target 0).
enum class SurfaceReads : uint8_t {
    Sampled,  ///< They may (xtframe-view, the editor's preview): not readable -> the intermediate.
    None,     ///< They never do (the C ABI GPU target): drawn into the surface whatever it offers.
};
struct SurfaceUsageChoice {
    WGPUTextureUsage usage = WGPUTextureUsage_RenderAttachment;
    bool readable = false;
    /// T-749: frames are drawn into the surface's texture itself -- readable, or SurfaceReads::None
    /// (then COPY_DST is never asked for). False: into the intermediate.
    bool direct = false;
};
SurfaceUsageChoice chooseSurfaceUsage(WGPUTextureUsage capsUsages,
                                      SurfaceReads reads = SurfaceReads::Sampled);

/// The native window a surface is made from.
struct WgpuNativeWindow {
    enum class Kind { None, Win32, X11, Wayland, MetalLayer, CocoaWindow };
    Kind kind = Kind::None;
    /// Win32: HINSTANCE (null = the process's own module); X11: Display*; Wayland: wl_display*.
    void* display = nullptr;
    /// Win32: HWND; Wayland: wl_surface*; MetalLayer: CAMetalLayer*; CocoaWindow: NSWindow* (a
    /// CAMetalLayer is made the layer of its content view; call on the main thread).
    void* window = nullptr;
    /// X11: the Window id.
    uint64_t x11Window = 0;
};

/// A single-sample `width` x `height` texture of `format` that can be drawn into, sampled and read
/// back (RENDER_ATTACHMENT | TEXTURE_BINDING | COPY_SRC): the offscreen target of a window-less
/// renderer. Null on failure. The caller destroys and releases it.
WGPUTexture createRenderTexture(WGPUDevice device, int width, int height, WGPUTextureFormat format,
                                const char* label = nullptr);

class WgpuWindowSurface {
public:
    WgpuWindowSurface() = default;
    ~WgpuWindowSurface();
    WgpuWindowSurface(const WgpuWindowSurface&) = delete;
    WgpuWindowSurface& operator=(const WgpuWindowSurface&) = delete;

    /// Creates the surface on `init.instance` and checks that `init.adapter` can present to it;
    /// picks the format, usage and present mode (file comment). Not configured yet. False and
    /// `error` when the window has no handle, the surface cannot be made or the adapter cannot
    /// present to it.
    /// T-749: `reads` as chooseSurfaceUsage() takes it.
    bool create(const WgpuInit& init, const WgpuNativeWindow& window, bool vsync, std::string& error,
                SurfaceReads reads = SurfaceReads::Sampled);
    /// T-758 tests: no window -- `texture` (not owned; alive while this object uses it) stands for every
    /// texture the surface would give, configured as chooseSurfaceUsage(its usage) says (the texture
    /// itself must have been made with at least that usage); present() moves the frame into it but
    /// presents nothing. A later call on the same device and format swaps the stand-in and keeps the
    /// intermediate (as a resized window gives textures of another size). False and `error` when
    /// `device` or `texture` is null. T-749: `reads` as create() takes it.
    bool createForTesting(WGPUDevice device, WGPUTexture texture, std::string& error,
                          SurfaceReads reads = SurfaceReads::Sampled);
    /// (Re)configures the surface for `device` at `width` x `height` pixels (sizes <= 0: 1).
    void configure(WGPUDevice device, int width, int height);
    /// Acquires the frame's texture (and a view of it). False when no texture can be had this frame
    /// (minimized, timeout, error): the caller skips the frame. Not readable: the frame's texture is
    /// the intermediate (made or remade at the surface texture's size); false when it cannot be made.
    bool acquire();
    /// Presents the acquired texture (nothing when none) and releases it; not readable: the
    /// intermediate is copied or drawn into the surface's texture first (one submit).
    void present();
    /// Releases the acquired texture without presenting it.
    void releaseFrame();
    /// Releases everything (unconfigures the surface). Safe to call twice.
    void release();

    bool valid() const { return m_surface != nullptr || m_testTexture != nullptr; }
    WGPUTextureFormat format() const { return m_format; }
    WGPUTextureUsage usage() const { return m_usage; }
    /// The surface was configured with TEXTURE_BINDING: frames are drawn into it directly. False: into
    /// the intermediate (file comment).
    bool surfaceReadable() const { return m_readable; }
    /// T-749: frames are drawn into the surface's texture itself (readable, or created with
    /// SurfaceReads::None); false: into the intermediate.
    bool drawsDirectly() const { return m_direct; }
    /// The frame's texture can be read back (COPY_SRC): always for the intermediate.
    bool frameCopyable() const;
    /// One line for the start-up notes: the usages offered and configured, and how frames reach the
    /// surface (directly, or through the intermediate by a copy or a full-screen pass).
    std::string usageNote() const;
    /// The frame's texture and its view (null outside acquire / present): the surface's, or the
    /// intermediate when the surface is not readable.
    WGPUTexture frameTexture() const { return m_frameTexture; }
    WGPUTextureView frameView() const { return m_frameView; }

private:
    bool ensureIntermediate(uint32_t width, uint32_t height);
    void releaseIntermediate();
    bool ensureBlit();
    void releaseBlit();
    /// The intermediate into the acquired surface texture: copyTextureToTexture, else a full-screen pass.
    void transferToSurface();

    WGPUSurface m_surface = nullptr;
    WGPUTexture m_testTexture = nullptr;  ///< createForTesting's stand-in (not owned)
    WGPUDevice m_device = nullptr;        ///< of the last configure (not owned)
    WGPUTextureFormat m_format = WGPUTextureFormat_BGRA8Unorm;
    WGPUTextureUsage m_offered = WGPUTextureUsage_None;  ///< the surface capabilities' usages
    WGPUTextureUsage m_usage = WGPUTextureUsage_RenderAttachment;
    bool m_readable = false;
    bool m_direct = false;  ///< T-749: SurfaceUsageChoice::direct
    WGPUPresentMode m_presentMode = WGPUPresentMode_Fifo;
    int m_width = 0, m_height = 0;
    WGPUTexture m_surfaceTexture = nullptr;   ///< the acquired surface texture (owned reference)
    WGPUTextureView m_surfaceView = nullptr;  ///< its view (owned; made only when readable or blitting)
    WGPUTexture m_frameTexture = nullptr;     ///< m_surfaceTexture or m_intermediate (not owned)
    WGPUTextureView m_frameView = nullptr;    ///< m_surfaceView or m_intermediateView (not owned)
    WGPUTexture m_intermediate = nullptr;     ///< not readable: what frames are drawn into (owned)
    WGPUTextureView m_intermediateView = nullptr;
    /// The full-screen pass of a surface with neither TEXTURE_BINDING nor COPY_DST (made on first use).
    WGPURenderPipeline m_blitPipeline = nullptr;
    WGPUBindGroupLayout m_blitLayout = nullptr;
};

}  // namespace xt::wgpu
