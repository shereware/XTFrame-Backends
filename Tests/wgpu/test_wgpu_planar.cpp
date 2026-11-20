// T-806 (Docs/design/23-host-integration.md §23.aj, Docs/tasks/T-806.md "測試"): planar external
// textures through xt_wgpu -- every leaf image path drawn from Y'CbCr planes (Code/Tools/Common/
// planar_pack.h's reference conversion of a known picture) next to the same picture drawn as an RGBA8
// texture with Shading::TextureStraight, the four matrix x range, the three sitings, alpha planes, the
// plane-bleed check (Tests/common/planar_scenes.h holds the cases and the tolerances' reasons), and an
// effect over a versioned planar texture served from the cache byte for byte.

#include <cstdint>
#include <string>
#include <vector>

#include "doctest.h"
#include "planar_scenes.h"
#include "test_wgpu_draw_util.h"

using namespace xt;
using namespace xt::wgpu::test;
namespace pt = xt::planar_test;

namespace {

class WgpuBackend : public pt::Backend {
public:
    WgpuBackend() : m_session(pt::kFrameW, pt::kFrameH) {}
    ~WgpuBackend() override {
        for (gpu::TextureHandle t : m_textures) {
            m_session.renderer.device()->destroyTexture(t);
        }
    }
    const char* name() const override { return "wgpu"; }
    uint32_t texture(int w, int h, bool r8, const uint8_t* data) override {
        const gpu::TextureHandle t = m_session.renderer.device()->createTexture(
            "t806 host", uint16_t(w), uint16_t(h), r8 ? gpu::TextureFormat::R8 : gpu::TextureFormat::Rgba8,
            data);
        m_textures.push_back(t);
        return t;
    }
    std::vector<uint8_t> render(const CommandList& list, AssetResolver& assets, const IrView& ir) override {
        return m_session.render(list, Color{0.1f, 0.1f, 0.12f, 1}, &assets, &ir);
    }
    std::string error() override { return m_session.error; }
    bool softwareAdapter() override { return xt::wgpu::test::cpuAdapter(); }  // T-812
    Session& session() { return m_session; }

private:
    Session m_session;
    std::vector<gpu::TextureHandle> m_textures;
};

}  // namespace

TEST_CASE("wgpu planar: every leaf image path matches the RGBA8 picture within the 8-bit 4:2:0 tolerance") {
    WgpuBackend backend;
    pt::runPixelCases(backend);
}

TEST_CASE("wgpu planar: no plane bleeds into its neighbour, at the texture's last column or a slice edge") {
    WgpuBackend backend;
    pt::runBleedCase(backend);
}

TEST_CASE("wgpu planar: an effect over a versioned planar texture is cached, byte for byte") {
    WgpuBackend backend;
    const pt::Picture picture = pt::smoothPicture();
    const tools::PackedPlanes planes = pt::planesOf(picture, AssetResolver::ExternalLayout::PackedYuv420);
    const uint32_t tex =
        backend.texture(planes.planes.textureWidth, planes.planes.textureHeight, true, planes.texels.data());
    pt::SceneResolver assets;
    assets.setExternalTexture("scene", tex, pt::kPicW, pt::kPicH);
    assets.setExternalTexturePlanes("scene", planes.planes);
    assets.setExternalTextureVersion("scene", 1);
    pt::Painted p;
    REQUIRE(
        pt::paint(R"X(<img src="engine://scene" style="position: absolute; left: 30px; top: 30px; width: 64px;
height: 43px; effect: shadow(4px 4px 3px 0px #000000cc)"/>)X",
                  assets, p));
    // The reference: a renderer that sees the list once (nothing cached).
    std::vector<uint8_t> reference;
    {
        WgpuBackend fresh;
        const uint32_t freshTex = fresh.texture(planes.planes.textureWidth, planes.planes.textureHeight, true,
                                                planes.texels.data());
        pt::SceneResolver freshAssets;
        freshAssets.setExternalTexture("scene", freshTex, pt::kPicW, pt::kPicH);
        freshAssets.setExternalTexturePlanes("scene", planes.planes);
        freshAssets.setExternalTextureVersion("scene", 1);
        reference = fresh.render(p.list, freshAssets, p.doc->ir());
        CHECK(fresh.session().renderer.stats().fxCacheHits == 0u);
    }
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i);
        const std::vector<uint8_t> px = backend.render(p.list, assets, p.doc->ir());
        CHECK_MESSAGE(backend.error().empty(), backend.error());
        if (i >= 2) {
            CHECK(backend.session().renderer.stats().fxCacheHits == 1u);
            CHECK(backend.session().renderer.stats().fxCacheMisses == 0u);
        }
        CHECK(pt::diff(px, reference).maxChannel == 0);
    }
    // The description changes, the version does not: drawn again (the binding moved).
    AssetResolver::ExternalPlanes other = planes.planes;
    other.matrix = AssetResolver::ExternalMatrix::Bt601;
    assets.setExternalTexturePlanes("scene", other);
    backend.render(p.list, assets, p.doc->ir());
    CHECK(backend.session().renderer.stats().fxCacheHits == 0u);
}
