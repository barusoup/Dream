// Temporal anti-aliasing, pass A: reproject, clip and blend.
//
// Reads the jittered scene colour, depth and velocity of this frame and the
// previous history, and writes the new history (rgb = resolved colour, a =
// motion estimate for the sharpen pass). Velocity is `prev_uv - curr_uv` in the
// renderer's bottom-up uv (`screen_uv`, uv.y = ndc.y * 0.5 + 0.5), which is
// also the texel space the offscreen targets are stored in.

struct TaaParams {
    size: vec4<f32>,    // render width, height, 1/width, 1/height
    jitter: vec4<f32>,  // xy: this frame's jitter in texels (xy of the sample offset); zw unused
    blend: vec4<f32>,   // x: history weight, y: variance gamma, z: sharpness, w: anti-flicker (0/1)
    flags: vec4<f32>,   // x: 1 when history is invalid (reset or first frame)
};

@group(0) @binding(0) var<uniform> taa: TaaParams;
@group(0) @binding(1) var scene_color: texture_2d<f32>;
@group(0) @binding(2) var scene_depth: texture_depth_2d;
@group(0) @binding(3) var scene_velocity: texture_2d<f32>;
@group(0) @binding(4) var history_in: texture_2d<f32>;
@group(0) @binding(5) var history_sampler: sampler;
@group(0) @binding(6) var history_out: texture_storage_2d<rgba16float, write>;

// Renderer velocity (bottom-up uv) to top-down texture uv.
const VELOCITY_TO_TEXTURE: vec2<f32> = vec2<f32>(1.0, 1.0);
// Blackman-Harris reconstruction support, in texels from the output pixel.
const RECON_RADIUS: f32 = 1.5;

fn rgb_to_ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        0.25 * c.r + 0.5 * c.g + 0.25 * c.b,
        0.5 * c.r - 0.5 * c.b,
        -0.25 * c.r + 0.5 * c.g - 0.25 * c.b,
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    let t = c.x - c.z;
    return vec3<f32>(t + c.y, c.x + c.z, t - c.y);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Blackman-Harris window over [-RECON_RADIUS, RECON_RADIUS] texels.
fn recon_weight(d: f32) -> f32 {
    let t = clamp(d / (2.0 * RECON_RADIUS) + 0.5, 0.0, 1.0);
    let a = 6.283185307 * t;
    return 0.35875 - 0.48829 * cos(a) + 0.14128 * cos(2.0 * a) - 0.01168 * cos(3.0 * a);
}

// Move q toward the box centre just far enough to lie inside the box. The
// segment from the centre to q is the one the history is clipped along.
fn clip_to_box(q: vec3<f32>, centre: vec3<f32>, half_size: vec3<f32>) -> vec3<f32> {
    let v = q - centre;
    let u = abs(v) / max(half_size, vec3<f32>(1e-6));
    let m = max(u.x, max(u.y, u.z));
    if m > 1.0 {
        return centre + v / m;
    }
    return q;
}

// Catmull-Rom history filter in five bilinear fetches: the four corners of the
// 4x4 footprint are dropped. Negative lobes are clamped to zero.
fn sample_history(uv: vec2<f32>) -> vec4<f32> {
    let size = taa.size.xy;
    let inv = taa.size.zw;
    let sample_pos = uv * size;
    let pos1 = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - pos1;
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let off12 = w2 / w12;
    let p0 = (pos1 - 1.0) * inv;
    let p3 = (pos1 + 2.0) * inv;
    let p12 = (pos1 + off12) * inv;

    let c_x0 = textureSampleLevel(history_in, history_sampler, vec2<f32>(p12.x, p0.y), 0.0);
    let c_0y = textureSampleLevel(history_in, history_sampler, vec2<f32>(p0.x, p12.y), 0.0);
    let c_xy = textureSampleLevel(history_in, history_sampler, vec2<f32>(p12.x, p12.y), 0.0);
    let c_3y = textureSampleLevel(history_in, history_sampler, vec2<f32>(p3.x, p12.y), 0.0);
    let c_x3 = textureSampleLevel(history_in, history_sampler, vec2<f32>(p12.x, p3.y), 0.0);

    let wx0 = w12.x * w0.y;
    let w0y = w0.x * w12.y;
    let wxy = w12.x * w12.y;
    let w3y = w3.x * w12.y;
    let wx3 = w12.x * w3.y;
    let total = wx0 + w0y + wxy + w3y + wx3;
    let sum = c_x0 * wx0 + c_0y * w0y + c_xy * wxy + c_3y * w3y + c_x3 * wx3;
    return max(sum / total, vec4<f32>(0.0));
}

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = taa.size.xy;
    if f32(gid.x) >= size.x || f32(gid.y) >= size.y {
        return;
    }
    let px = vec2<i32>(gid.xy);
    let max_px = vec2<i32>(size) - vec2<i32>(1);
    // Texel q shows the unjittered point at q + 0.5 + jitter. The offscreen
    // targets are stored bottom-up (`uv.y = ndc.y * 0.5 + 0.5`, see
    // `crate::temporal`), so x and y share one sign.
    let jitter = vec2<f32>(taa.jitter.x, taa.jitter.y);

    // 1. Dilate motion: the velocity of the nearest-depth texel in a 5-tap cross.
    var nearest = px;
    var nearest_depth = 2.0;
    for (var i = 0u; i < 5u; i++) {
        var o = vec2<i32>(0);
        if i == 1u { o = vec2<i32>(1, 0); }
        if i == 2u { o = vec2<i32>(-1, 0); }
        if i == 3u { o = vec2<i32>(0, 1); }
        if i == 4u { o = vec2<i32>(0, -1); }
        let q = clamp(px + o, vec2<i32>(0), max_px);
        let d = textureLoad(scene_depth, q, 0);
        if d < nearest_depth {
            nearest_depth = d;
            nearest = q;
        }
    }
    let velocity = textureLoad(scene_velocity, nearest, 0).xy * VELOCITY_TO_TEXTURE;

    // 2. Reproject: history is stored reconstructed at pixel centres (no
    // jitter), so this pixel's history lies one motion away from its own
    // centre. Only the *velocity* comes from the nearest-depth texel; using
    // that texel's position too would shift edge pixels by a texel.
    let hist_uv = (vec2<f32>(px) + 0.5) / size + velocity;
    let speed = length(velocity * size);
    let invalid = taa.flags.x > 0.5;
    let inside = all(hist_uv >= vec2<f32>(0.0)) && all(hist_uv <= vec2<f32>(1.0));

    // 3. Current 3x3: Blackman-Harris reconstruction at the pixel centre, and
    // the neighbourhood statistics in YCoCg.
    var recon_sum = vec3<f32>(0.0);
    var recon_total = 0.0;
    var box_min = vec3<f32>(1e30);
    var box_max = vec3<f32>(-1e30);
    var mean = vec3<f32>(0.0);
    var moment = vec3<f32>(0.0);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = clamp(px + vec2<i32>(dx, dy), vec2<i32>(0), max_px);
            let c = textureLoad(scene_color, q, 0).rgb;
            // The texel's sample sits at (dx, dy) + jitter from the pixel centre.
            let d = vec2<f32>(f32(dx), f32(dy)) + jitter;
            let w = recon_weight(d.x) * recon_weight(d.y);
            recon_sum += c * w;
            recon_total += w;
            let ycc = rgb_to_ycocg(c);
            box_min = min(box_min, ycc);
            box_max = max(box_max, ycc);
            mean += ycc;
            moment += ycc * ycc;
        }
    }
    let current = recon_sum / max(recon_total, 1e-6);
    mean = mean / 9.0;
    let sigma = sqrt(max(moment / 9.0 - mean * mean, vec3<f32>(0.0)));
    let gamma = taa.blend.y;
    let lo = max(mean - gamma * sigma, box_min);
    let hi = min(mean + gamma * sigma, box_max);

    // With no usable history (first frame, reset, or the reprojection left
    // the screen) the current frame stands alone: alpha = 1 and history unused.
    var history = current;
    var alpha = 1.0;
    if !invalid && inside {
        // 4. Clip the history to the box along the line to its centre.
        let raw = sample_history(hist_uv);
        let raw_ycc = rgb_to_ycocg(raw.rgb);
        let centre = 0.5 * (lo + hi);
        let half_size = 0.5 * (hi - lo);
        history = ycocg_to_rgb(clip_to_box(raw_ycc, centre, half_size));

        // 5. Blend weight: static pixels keep their history, moving ones take
        // the new frame sooner, and (anti-flicker) history whose luma sits near
        // the box edge is trusted less.
        alpha = 1.0 - taa.blend.x;
        let motion = smoothstep(0.5, 8.0, speed) * 0.75;
        alpha = mix(alpha, 1.0, motion);
        if taa.blend.w > 0.5 {
            let span = max(hi.x - lo.x, 1e-4);
            let edge = clamp(min(raw_ycc.x - lo.x, hi.x - raw_ycc.x) / span, 0.0, 0.5);
            alpha = alpha * mix(0.25, 1.0, clamp(edge * 4.0, 0.0, 1.0));
        }
    }

    // 6. Karis weighting: blend in a luminance-compressed space so that one
    // bright sample does not dominate the average.
    let w_cur = alpha / (1.0 + luma(max(current, vec3<f32>(0.0))));
    let w_hist = (1.0 - alpha) / (1.0 + luma(max(history, vec3<f32>(0.0))));
    let out_rgb = max((current * w_cur + history * w_hist) / max(w_cur + w_hist, 1e-6),
        vec3<f32>(0.0));

    // 7. Motion estimate for the sharpen pass.
    textureStore(history_out, px, vec4<f32>(out_rgb, saturate(speed * 0.25)));
}
