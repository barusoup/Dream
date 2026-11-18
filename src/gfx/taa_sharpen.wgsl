// Temporal anti-aliasing, pass B: contrast-adaptive sharpening.
//
// Reads the resolved history (never written back to it) and writes the image
// the blit samples. The filter runs on a 5-tap cross in the reversible
// tonemapped space c / (1 + max c). The negative lobe is scaled by the local
// headroom (min against 1 - max, and against the local minimum), so it cannot
// push a channel below zero, above one, or past the local range, which keeps
// it from ringing. The shaping follows the contrast-adaptive sharpening idea,
// written here from scratch.

struct TaaParams {
    size: vec4<f32>,    // render width, height, 1/width, 1/height
    jitter: vec4<f32>,  // unused by this pass
    blend: vec4<f32>,   // z: sharpness (0..1)
    flags: vec4<f32>,   // unused by this pass
};

@group(0) @binding(0) var<uniform> taa: TaaParams;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var sharpened: texture_storage_2d<rgba16float, write>;

fn tone(c: vec3<f32>) -> vec3<f32> {
    let m = max(max(c.r, c.g), c.b);
    return c / (1.0 + max(m, 0.0));
}

fn untone(t: vec3<f32>) -> vec3<f32> {
    let m = clamp(max(max(t.r, t.g), t.b), 0.0, 0.999);
    return t / (1.0 - m);
}

@compute @workgroup_size(8, 8)
fn sharpen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = taa.size.xy;
    if f32(gid.x) >= size.x || f32(gid.y) >= size.y {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let max_px = vec2<i32>(size) - vec2<i32>(1);

    let centre = textureLoad(history, px, 0);
    let n = tone(textureLoad(history, clamp(px + vec2<i32>(0, -1), vec2<i32>(0), max_px), 0).rgb);
    let s = tone(textureLoad(history, clamp(px + vec2<i32>(0, 1), vec2<i32>(0), max_px), 0).rgb);
    let e = tone(textureLoad(history, clamp(px + vec2<i32>(1, 0), vec2<i32>(0), max_px), 0).rgb);
    let w = tone(textureLoad(history, clamp(px + vec2<i32>(-1, 0), vec2<i32>(0), max_px), 0).rgb);
    let c = tone(centre.rgb);

    // Local range of the cross, per channel.
    let lo = min(min(min(n, s), min(e, w)), c);
    let hi = max(max(max(n, s), max(e, w)), c);
    // Headroom to the local range and to the tonemapped ceiling of 1.
    let amp = sqrt(clamp(min(lo, 1.0 - hi) / max(hi, vec3<f32>(1e-4)), vec3<f32>(0.0), vec3<f32>(1.0)));

    // Motion-reconstructed pixels (history alpha) take a little more sharpening.
    let strength = clamp(taa.blend.z * mix(1.0, 1.3, saturate(centre.a)), 0.0, 1.0);
    let peak = -1.0 / mix(8.0, 5.0, strength);
    let lobe = amp * peak;
    let out_t = (c + lobe * (n + s + e + w)) / (1.0 + 4.0 * lobe);
    let out_rgb = untone(clamp(out_t, vec3<f32>(0.0), vec3<f32>(0.999)));
    textureStore(sharpened, px, vec4<f32>(out_rgb, 1.0));
}
