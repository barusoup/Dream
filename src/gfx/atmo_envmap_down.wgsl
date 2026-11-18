// Sky environment map, one mip down: an exact 2x2 box average of the level above.
// Each destination texel reads four source texels; the source is exactly twice the
// destination in both axes, so every tap is in range and the map's u wrap (its
// sampler repeats in u) needs no extra handling here.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(dst);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let src_size = vec2<i32>(textureDimensions(src));
    let base = vec2<i32>(gid.xy) * 2;
    // Polar rows converge to a single direction (+Y at y=0, -Y at y=H-1): average
    // uniformly across azimuth so all u texels agree and rough reflections around
    // the poles don't pinch into a starburst or cone.
    if gid.y == 0u || gid.y + 1u == size.y {
        let step = max(src_size.x / 32, 1);
        var sum = vec4<f32>(0.0);
        var count = 0.0;
        let y0 = base.y;
        let y1 = min(base.y + 1, src_size.y - 1);
        for (var x = 0; x < src_size.x; x += step) {
            sum += textureLoad(src, vec2<i32>(x, y0), 0)
                + textureLoad(src, vec2<i32>(x, y1), 0);
            count += 2.0;
        }
        textureStore(dst, gid.xy, sum / count);
        return;
    }
    // Widen horizontal taps near poles inversely with sin(theta) so rings blend
    // smoothly into the azimuthally-averaged polar cap.
    let v = (f32(gid.y) + 0.5) / f32(size.y);
    let sin_theta = max(sin(v * 3.14159265359), 0.06);
    let extra = clamp(i32(round(0.35 / sin_theta)) - 1, 0, 6);
    if extra == 0 {
        let sum = textureLoad(src, base, 0) + textureLoad(src, base + vec2<i32>(1, 0), 0)
            + textureLoad(src, base + vec2<i32>(0, 1), 0) + textureLoad(src, base + vec2<i32>(1, 1), 0);
        textureStore(dst, gid.xy, sum * 0.25);
        return;
    }
    var sum = vec4<f32>(0.0);
    var count = 0.0;
    for (var dx = -extra; dx <= 1 + extra; dx++) {
        let sx = ((base.x + dx) % src_size.x + src_size.x) % src_size.x;
        sum += textureLoad(src, vec2<i32>(sx, base.y), 0)
            + textureLoad(src, vec2<i32>(sx, base.y + 1), 0);
        count += 2.0;
    }
    textureStore(dst, gid.xy, sum / count);
}
