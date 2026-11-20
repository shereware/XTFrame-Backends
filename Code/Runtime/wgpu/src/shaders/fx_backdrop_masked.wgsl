// EffectProgram::BackdropMasked (T-729, Docs/design/10-backends.md section 10.18 B). Mirrors gl_shaders.h
// kFxBackdropMaskedFrag: t0 = R' (the content with its shadow-type effects), t1 = the blurred backdrop,
// t2 = the content alone, whose alpha masks it.
//
// T-764: gpu_frame.h BackdropParams -- pf(1) topIsTexture, pf(2) maskIsTexture, pf(5..8) topUv,
// pf(9..12) maskUv, pf(13) topSampler, pf(14) maskSampler (the handles, pf(3..4), are bound by the device).
// A one-float block is not uploaded: the zero block reads as two target inputs, the program of before.

// An input of the program: a target sampled at the window's uv as before, or a cached texture at
// uvT = uv * scale + offset, transparent outside [0,1)^2; Sampler::PointClamp (0) = the texel under uvT,
// Sampler::LinearClamp (1) = bilinear with the command's (LinearClamp) sampler.
fn backdropInput(t: texture_2d<f32>, uv: vec2f, isTexture: f32, uvAt: u32, sampling: f32) -> vec4f {
    if (isTexture == 0.0) {
        return textureSampleLevel(t, smp, uv, 0.0);
    }
    let p = uv * vec2f(pf(uvAt), pf(uvAt + 1u)) + vec2f(pf(uvAt + 2u), pf(uvAt + 3u));
    if (p.x < 0.0 || p.y < 0.0 || p.x >= 1.0 || p.y >= 1.0) {
        return vec4f(0.0);
    }
    if (sampling == 0.0) {
        let size = vec2i(textureDimensions(t));
        return textureLoad(t, min(vec2i(floor(p * vec2f(size))), size - vec2i(1)), 0);
    }
    return textureSampleLevel(t, smp, p, 0.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let top = backdropInput(t0, in.uv, pf(1u), 5u, pf(13u));
    let mask = backdropInput(t2, in.uv, pf(2u), 9u, pf(14u));
    let bottom = tex1(in.uv) * mask.a;
    let inv = 1.0 - top.a;
    return vec4f(top.rgb + bottom.rgb * inv, top.a + bottom.a * inv);
}
