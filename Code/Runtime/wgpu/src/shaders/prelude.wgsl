// xt_wgpu shader prelude (T-438, Docs/design/10-backends.md section 10.10.2): prepended to every
// fragment file of this directory. Bindings, the shared vertex stages and the parameter accessor.
// Mirrors Code/Runtime/gl/src/gl_shaders.h: vs_main = kMainVert, vs_quad = kQuadVert, vs_particle =
// kParticleVert. WebGPU render targets are top-down like uploaded textures (gpu_frame.h "Texture
// sampling"), so vs_quad does not flip v and no shader here reconstructs a flipped row.
//
// group 0: the pass constants (surface px -> NDC: the three columns of GL's uNdc), dynamic offset;
// group 1: binding 0 = the command's parameter block (gpu_frame.h's *Params structs as raw floats),
//          dynamic offset, read with pf(index); binding 1 = the command's analytic clip block
//          (T-718: gpu_frame.h AnalyticClipParams at DrawCmd::clipParamIndex; the zero block, count
//          0, for a command without one), dynamic offset, read by clipCoverage();
// group 2: t0 = the command's texture or src target, t1 = aux, t2 = the GradientMask LUT,
//          smp = the command's sampler.

struct PassConsts {
    m0: vec4f,
    m1: vec4f,
    m2: vec4f,
}
@group(0) @binding(0) var<uniform> uPass: PassConsts;

struct ParamBlock {
    v: array<vec4f, 32>,
}
@group(1) @binding(0) var<uniform> uParams: ParamBlock;

// gpu_frame.h AnalyticClipParams: the slot count, then per slot four vec4 -- localFromDevice
// (a, b, c, d), (e, f, -, -), the rectangle (x, y, w, h) and its radii (tl, tr, br, bl).
struct ClipBlock {
    head: vec4u,
    slots: array<vec4f, 12>,
}
@group(1) @binding(1) var<uniform> uClip: ClipBlock;

@group(2) @binding(0) var t0: texture_2d<f32>;
@group(2) @binding(1) var t1: texture_2d<f32>;
@group(2) @binding(2) var t2: texture_2d<f32>;
@group(2) @binding(3) var smp: sampler;

// Float `i` of the command's parameter block (&GpuFrame::params[cmd.paramIndex]).
fn pf(i: u32) -> f32 {
    return uParams.v[i >> 2u][i & 3u];
}

// A parameter the GL device uploads as an int (static_cast<int>: truncation toward zero).
fn pi(i: u32) -> i32 {
    return i32(pf(i));
}

// uNdc * vec3(xy, 1).
fn ndc(xy: vec2f) -> vec2f {
    return uPass.m0.xy * xy.x + uPass.m1.xy * xy.y + uPass.m2.xy;
}

// textureSampleLevel at level 0 everywhere: the GLSL samples inside non-uniform control flow, which
// WGSL allows only for the explicit-level form (every texture has one level, so it is the same).
fn tex0(uv: vec2f) -> vec4f {
    return textureSampleLevel(t0, smp, uv, 0.0);
}
fn tex1(uv: vec2f) -> vec4f {
    return textureSampleLevel(t1, smp, uv, 0.0);
}

// px (T-718): the fragment's position in surface pixels (gpu_frame.h "Coordinates"), interpolated
// linearly in screen space, so it is exact for projected (w != 1) vertices too.
struct VsOut {
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
    @location(1) color: vec4f,
    @location(2) @interpolate(linear) px: vec2f,
}

// T-718 (Docs/design/10-backends.md section 10.16 W1 item 2): the coverage of clip slot `i` at the
// surface pixel position `px`: clamp(0.5 - d, 0, 1), d = the signed distance to the rounded
// rectangle's boundary in device pixels -- the local signed distance over the length of its
// gradient taken back to device space (M^T g, M = the 2x2 of localFromDevice).
fn clipSlotCoverage(i: u32, px: vec2f) -> f32 {
    let m = uClip.slots[i * 4u];
    let t = uClip.slots[i * 4u + 1u];
    let rect = uClip.slots[i * 4u + 2u];
    let radii = uClip.slots[i * 4u + 3u];
    let p = vec2f(m.x * px.x + m.z * px.y + t.x, m.y * px.x + m.w * px.y + t.y);
    let half = rect.zw * 0.5;
    let q0 = p - (rect.xy + half);
    let right = q0.x > 0.0;
    let r = select(select(radii.x, radii.y, right), select(radii.w, radii.z, right), q0.y > 0.0);
    let s = vec2f(select(-1.0, 1.0, right), select(-1.0, 1.0, q0.y > 0.0));
    let q = abs(q0) - half + vec2f(r);
    var d: f32;
    var g: vec2f;
    if (q.x > 0.0 && q.y > 0.0) {
        let l = length(q);
        d = l - r;
        g = q / l * s;
    } else if (q.x > q.y) {
        d = q.x - r;
        g = vec2f(s.x, 0.0);
    } else {
        d = q.y - r;
        g = vec2f(0.0, s.y);
    }
    let gd = vec2f(m.x * g.x + m.y * g.y, m.z * g.x + m.w * g.y);
    return clamp(0.5 - d / max(length(gd), 1e-20), 0.0, 1.0);
}

// The product of the command's analytic clip coverages at `px`; exactly 1 without a clip (count 0),
// so a leaf program's output times this is its output of before.
fn clipCoverage(px: vec2f) -> f32 {
    var c = 1.0;
    let n = min(uClip.head.x, 3u);
    for (var i = 0u; i < n; i = i + 1u) {
        c = c * clipSlotCoverage(i, px);
    }
    return c;
}

// kMainVert: posW = (x*w, y*w, w), divided to NDC and re-multiplied by w (19 section 19.7).
@vertex
fn vs_main(@location(0) posW: vec3f, @location(1) uv: vec2f, @location(2) color: vec4f) -> VsOut {
    var o: VsOut;
    let p = ndc(posW.xy / posW.z);
    o.pos = vec4f(p * posW.z, 0.0, posW.z);
    o.uv = uv;
    o.color = color;
    o.px = posW.xy / posW.z;
    return o;
}

// kQuadVert without its v flip (see the file comment).
@vertex
fn vs_quad(@location(0) posW: vec3f, @location(1) uv: vec2f, @location(2) color: vec4f) -> VsOut {
    var o: VsOut;
    let p = ndc(posW.xy / posW.z);
    o.pos = vec4f(p * posW.z, 0.0, posW.z);
    o.uv = uv;
    o.color = vec4f(1.0);
    o.px = posW.xy / posW.z;
    return o;
}

// kParticleVert: six vertices per instance (two triangles); ParticleParams::xf at pf(0..5).
@vertex
fn vs_particle(@builtin(vertex_index) vi: u32, @location(0) centerSize: vec4f, @location(1) rot: f32,
               @location(2) color: vec4f, @location(3) uvRect: vec4f) -> VsOut {
    var corners = array<vec2f, 6>(vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
                                  vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(0.0, 1.0));
    let corner = corners[vi];
    let d = (corner - 0.5) * centerSize.zw;
    let c = cos(rot);
    let s = sin(rot);
    let local = centerSize.xy + vec2f(c * d.x - s * d.y, s * d.x + c * d.y);
    let px = vec2f(pf(0u) * local.x + pf(2u) * local.y + pf(4u), pf(1u) * local.x + pf(3u) * local.y + pf(5u));
    var o: VsOut;
    o.pos = vec4f(ndc(px), 0.0, 1.0);
    o.uv = mix(uvRect.xy, uvRect.zw, corner);
    o.color = color;
    o.px = px;
    return o;
}

// GL's smoothstepGuarded (kFxShineFrag, kFxOutlineFrag).
fn smoothstepGuarded(e0: f32, e1: f32, x: f32) -> f32 {
    if (e1 <= e0) {
        return select(1.0, 0.0, x < e0);
    }
    let t = clamp((x - e0) / (e1 - e0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn insideUnit(uv: vec2f) -> bool {
    return uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0;
}

fn lum(c: vec3f) -> f32 {
    return dot(c, vec3f(0.3, 0.59, 0.11));
}
