@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

// Per-frame blit options (push constants, fragment stage). With
// tonemap == 0 and dither == 0 both entry points return the source unchanged.
struct BlitPush {
    tonemap: u32,   // 0 none, 1 ACES, 2 AgX, 3 Khronos PBR Neutral
    exposure: f32,  // multiplies linear colour before the curve
    dither: u32,    // nonzero: add +-0.5 LSB noise before quantisation
    pad: u32,
};
var<push_constant> blit: BlitPush;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(p[i], 0.0, 1.0);
    o.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return o;
}

// Narkowicz's fitted ACES filmic curve (not the full RRT + ODT matrices).
fn tone_aces(x: vec3<f32>) -> vec3<f32> {
    let y = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
    return clamp(y, vec3<f32>(0.0), vec3<f32>(1.0));
}

// Minimal AgX with the base look: inset, log2 encode, sigmoid contrast fit,
// outset, then a 2.2 power to linear.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x
        + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn tone_agx(x: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3<f32>(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3<f32>(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3<f32>(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3<f32>(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = inset * x;
    v = clamp(log2(max(v, vec3<f32>(1e-10))), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    v = agx_contrast(v);
    v = max(outset * v, vec3<f32>(0.0));
    return pow(v, vec3<f32>(2.2));
}

// Khronos PBR Neutral (reference implementation, version 1.0).
fn tone_neutral(color_in: vec3<f32>) -> vec3<f32> {
    let start_compression = 0.8 - 0.04;
    let desaturation = 0.15;
    var color = color_in;
    let x = min(color.r, min(color.g, color.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    color = color - offset;
    let peak = max(color.r, max(color.g, color.b));
    if peak < start_compression {
        return color;
    }
    let d = 1.0 - start_compression;
    let new_peak = 1.0 - d * d / (peak + d - start_compression);
    color = color * (new_peak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, vec3<f32>(new_peak), vec3<f32>(g));
}

// Exposure and the selected curve, in linear light.
fn tone_map(c: vec3<f32>) -> vec3<f32> {
    let x = max(c * blit.exposure, vec3<f32>(0.0));
    if blit.tonemap == 1u {
        return tone_aces(x);
    }
    if blit.tonemap == 2u {
        return tone_agx(x);
    }
    return tone_neutral(x);
}

// Interleaved gradient noise in [-0.5, 0.5): one LSB of an 8-bit output, centred.
fn dither_offset(pos: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(pos, vec2<f32>(0.06711056, 0.00583715)))) - 0.5;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, samp, in.uv);
    if blit.tonemap == 0u && blit.dither == 0u {
        return c;
    }
    var rgb = c.rgb;
    if blit.tonemap != 0u {
        rgb = tone_map(rgb);
    }
    if blit.dither != 0u {
        rgb = rgb + vec3<f32>(dither_offset(in.pos.xy) / 255.0);
    }
    return vec4<f32>(rgb, c.a);
}

// sRGB encode of linear values (the piecewise OETF).
fn srgb_encode(c: vec3<f32>) -> vec3<f32> {
    let low = c * 12.92;
    let high = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - vec3<f32>(0.055);
    return select(high, low, c <= vec3<f32>(0.0031308));
}

/// Gamma-blend display target (`UiBlend::Gamma`): the display image is
/// UNORM holding sRGB-encoded values, so the linear scene is encoded here.
@fragment
fn fs_gamma(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(src_tex, samp, in.uv);
    if blit.tonemap == 0u && blit.dither == 0u {
        return vec4<f32>(srgb_encode(c.rgb), c.a);
    }
    var rgb = c.rgb;
    if blit.tonemap != 0u {
        rgb = tone_map(rgb);
    }
    var encoded = srgb_encode(rgb);
    if blit.dither != 0u {
        encoded = encoded + vec3<f32>(dither_offset(in.pos.xy) / 255.0);
    }
    return vec4<f32>(encoded, c.a);
}
