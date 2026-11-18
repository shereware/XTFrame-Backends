// Shading::Texture. Mirrors gl_shaders.h kTextureFrag.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    return tex0(in.uv) * in.color * clipCoverage(in.px);  // T-718: 1 without an analytic clip
}
