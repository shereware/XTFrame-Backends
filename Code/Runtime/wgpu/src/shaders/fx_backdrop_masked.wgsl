// EffectProgram::BackdropMasked (T-729, Docs/design/10-backends.md section 10.18 B; BackdropParams is
// read by the device, not here). Mirrors gl_shaders.h kFxBackdropMaskedFrag: t0 = R' (the content with
// its shadow-type effects), t1 = the blurred backdrop, t2 = the content alone, whose alpha masks it.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
    let top = tex0(in.uv);
    let bottom = tex1(in.uv) * textureSampleLevel(t2, smp, in.uv, 0.0).a;
    let inv = 1.0 - top.a;
    return vec4f(top.rgb + bottom.rgb * inv, top.a + bottom.a * inv);
}
