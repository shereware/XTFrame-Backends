// T-806 (Docs/design/23-host-integration.md section 23.aj): Shading::TexturePlanar -- Y'CbCr 4:2:0
// planes packed in one R8 texture (t0), converted to premultiplied RGB while sampling. gpu_frame.h
// PlanarParams (12 vec4): uv[4] (scale, offset) at vec4 0..3, clampRect[4] at 4..7, the matrix rows
// at 8..10, (hasAlpha, premultiplied, -, -) at 11. Each plane is sampled at its own uv clamped to its
// rectangle inset by half a texel, so a bilinear sample never reads the neighbouring plane. No branch
// on the matrix, the range or the siting (they are in the numbers). Mirrors gl_shaders.h
// kTexturePlanarFrag.
fn planeSample(p: u32, uv: vec2f) -> f32 {
    let m = uParams.v[p];
    let r = uParams.v[4u + p];
    let t = clamp(uv * m.xy + m.zw, r.xy, r.zw);
    return textureSampleLevel(t0, smp, t, 0.0).r;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let yuv1 = vec4f(planeSample(0u, in.uv), planeSample(1u, in.uv), planeSample(2u, in.uv), 1.0);
    var rgb = clamp(vec3f(dot(uParams.v[8], yuv1), dot(uParams.v[9], yuv1), dot(uParams.v[10], yuv1)),
                    vec3f(0.0), vec3f(1.0));
    let flags = uParams.v[11];
    var a = 1.0;
    if (flags.x != 0.0) {
        a = planeSample(3u, in.uv);
        if (flags.y == 0.0) {
            rgb = rgb * a;
        }
    }
    return vec4f(rgb, a) * in.color * clipCoverage(in.px);  // T-718: 1 without an analytic clip
}
