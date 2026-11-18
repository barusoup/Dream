// Sky environment map, mip 0: the sky as the sky pass draws it, one texel per
// direction of an equirectangular map (u = atan2(z, x) / 2pi + 1/2, v = acos(y) / pi,
// so v = 0 is the zenith). Upper hemisphere: sky-view LUT radiance plus the night
// floor, with the cloud panorama composited over it when clouds are on. Lower
// hemisphere: the horizon sky of the same azimuth, blended into the ground radiance
// over the first 3 degrees below the horizon. Prepended with atmo_common.wgsl and
// cloud_common.wgsl at load (build.rs and atmosphere.rs compose it the same way).
// The inverse of `sky_env_uv` in lighting.wgsl.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var<uniform> env: EnvMapParams;
@group(0) @binding(2) var skyview: texture_2d<f32>;
@group(0) @binding(3) var lut_sampler: sampler;
@group(0) @binding(4) var cloud_panorama: texture_2d<f32>;
@group(0) @binding(5) var cloud_samp: sampler;
@group(0) @binding(6) var out_map: texture_storage_2d<rgba16float, write>;

// Ground radiance seen below the horizon (rgb, from the driven environment).
struct EnvMapParams {
    ground: vec4<f32>,
};

// Width of the blend from the horizon sky into the ground, radians (3 degrees).
const ENV_HORIZON_BLEND: f32 = 0.05235988;

// Direction (unit, y up) of a map uv: the inverse of the lighting's `sky_env_uv`.
fn atmo_env_dir(uv: vec2<f32>) -> vec3<f32> {
    let phi = (uv.x - 0.5) * 2.0 * ATMO_PI;
    let theta = uv.y * ATMO_PI;
    let st = sin(theta);
    return vec3<f32>(st * cos(phi), cos(theta), st * sin(phi));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(out_map);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(size);
    let dir = atmo_env_dir(uv);
    let sun = atmo.sun_dir.xyz;
    var c: vec3<f32>;
    if dir.y >= 0.0 {
        c = atmo_sky_view_radiance(atmo, skyview, lut_sampler, dir, sun) + atmo_night(atmo);
        if atmo.sun_irradiance.w > 0.5 {
            c = cloud_composite(c, dir, cloud_panorama, cloud_samp);
        }
    } else {
        // The horizon sky of this azimuth, from the same LUT the sky pass uses.
        let hd = vec3<f32>(dir.x, 0.0, dir.z);
        let hl = length(hd);
        var az = vec3<f32>(1.0, 0.0, 0.0);
        if hl > 1e-6 {
            az = hd / hl;
        }
        let horizon = atmo_sky_view_radiance(atmo, skyview, lut_sampler,
            normalize(vec3<f32>(az.x, 0.0005, az.z)), sun) + atmo_night(atmo);
        let below = asin(clamp(-dir.y, 0.0, 1.0));
        c = mix(horizon, env.ground.rgb, smoothstep(0.0, ENV_HORIZON_BLEND, below));
    }
    textureStore(out_map, gid.xy, vec4<f32>(c, 1.0));
}
