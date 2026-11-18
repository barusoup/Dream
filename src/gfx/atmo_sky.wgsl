// Sky pass: fullscreen draw where the scene has no geometry (depth == 1, tested
// EQUAL in the pass). Writes scene colour (replace) and velocity. Prepended
// with atmo_common.wgsl at load. The clouds are not drawn here: the per-pixel
// cloud composite (cloud_view_composite.wgsl) runs after this pass.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var skyview: texture_2d<f32>;
@group(0) @binding(2) var tlut: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;

// Brightest sun-disk radiance written, to keep the HDR buffer free of fireflies.
const ATMO_SUN_DISK_MAX: f32 = 256.0;

struct SkyVsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> SkyVsOut {
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    let c = corners[i];
    var out: SkyVsOut;
    out.pos = vec4<f32>(c, 1.0, 1.0);
    out.ndc = c;
    return out;
}

struct SkyFsOut {
    @location(0) colour: vec4<f32>,
    @location(1) velocity: vec2<f32>,
};

@fragment
fn fs_main(in: SkyVsOut) -> SkyFsOut {
    let clip = atmo.inv_view_proj * vec4<f32>(in.ndc, 0.5, 1.0);
    let world = clip.xyz / clip.w;
    let dir = normalize(world - atmo.cam_pos.xyz);
    let sun = atmo.sun_dir.xyz;
    // Below the horizon, the model's planet ground would show as a flat band
    // wherever geometry ends. Unless `show_ground` is set (ground.w), continue the
    // horizon colour of the same azimuth downward as haze: full strength at the
    // horizon, 0.6 times from 6 degrees below. The LUTs are unchanged.
    var sdir = dir;
    var haze = 1.0;
    if atmo.ground.w < 0.5 && dir.y < 0.0 {
        let hd = vec3<f32>(dir.x, 0.0, dir.z);
        let hl = length(hd);
        var az = vec3<f32>(1.0, 0.0, 0.0);
        if hl > 1e-6 {
            az = hd / hl;
        }
        sdir = normalize(vec3<f32>(az.x, 0.0005, az.z));
        let below = asin(clamp(-dir.y, 0.0, 1.0));
        haze = mix(1.0, 0.6, smoothstep(0.0, 0.10471976, below));
    }
    var radiance = atmo_sky_view_radiance(atmo, skyview, lut_sampler, sdir, sun) * haze
        + atmo_night(atmo);
    if atmo.camera.y > 0.5 {
        let ang = acos(clamp(dot(dir, sun), -1.0, 1.0));
        let rad = atmo.camera.z;
        if ang < rad {
            let x = ang / rad;
            let limb = 1.0 - 0.6 * (1.0 - sqrt(max(1.0 - x * x, 0.0)));
            let t = atmo_transmittance(atmo, tlut, lut_sampler, atmo.camera.x, sun.y);
            let disk = atmo.sun_irradiance.rgb * t * limb / (ATMO_PI * rad * rad);
            radiance += min(disk, vec3<f32>(ATMO_SUN_DISK_MAX));
        }
    }
    // Motion of a point at infinity under camera rotation only.
    let prev = atmo.prev_view_proj * vec4<f32>(dir, 1.0);
    let curr = atmo.view_proj * vec4<f32>(dir, 1.0);
    let uv_prev = prev.xy / prev.w * 0.5 + 0.5;
    let uv_curr = curr.xy / curr.w * 0.5 + 0.5;
    var out: SkyFsOut;
    out.colour = vec4<f32>(radiance, 1.0);
    out.velocity = uv_prev - uv_curr;
    return out;
}
