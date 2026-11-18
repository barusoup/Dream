// Multiple-scattering LUT pass (psi, per unit sun irradiance). Prepended with
// atmo_common.wgsl at load.

@group(0) @binding(0) var<uniform> atmo: AtmoParams;
@group(0) @binding(1) var tlut: texture_2d<f32>;
@group(0) @binding(2) var lut_sampler: sampler;
@group(0) @binding(3) var multiscatter_out: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<u32>(32u, 32u);
    if gid.x >= size.x || gid.y >= size.y {
        return;
    }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let rm = atmo_ms_params(atmo, uv);
    let r = rm.x;
    let mu_s = rm.y;
    let sun = vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    let origin = vec3<f32>(0.0, r, 0.0);
    let golden = ATMO_PI * (3.0 - sqrt(5.0));
    var acc_l = vec3<f32>(0.0);
    var acc_f = vec3<f32>(0.0);
    let n = f32(ATMO_MS_DIRECTIONS);
    for (var i = 0u; i < ATMO_MS_DIRECTIONS; i++) {
        let z = 1.0 - (2.0 * f32(i) + 1.0) / n;
        let s = sqrt(max(1.0 - z * z, 0.0));
        let phi = f32(i) * golden;
        let dir = vec3<f32>(s * cos(phi), z, s * sin(phi));
        var tmax = atmo_top_distance(atmo, r, dir.y);
        if atmo_hits_ground(atmo, r, dir.y) {
            tmax = atmo_ground_distance(atmo, r, dir.y);
        }
        let dt = tmax / f32(ATMO_MS_STEPS);
        var t_acc = vec3<f32>(1.0);
        var l = vec3<f32>(0.0);
        var f = vec3<f32>(0.0);
        for (var j = 0u; j < ATMO_MS_STEPS; j++) {
            let t = (f32(j) + 0.5) * dt;
            let pos = origin + dir * t;
            let rr = length(pos);
            let m = atmo_medium(atmo, rr - atmo.planet.x);
            let mu_st = dot(pos, sun) / rr;
            let sun_t = atmo_transmittance(atmo, tlut, lut_sampler, rr, mu_st);
            let sint = (vec3<f32>(1.0) - exp(-m.ext * dt)) / m.ext;
            let scat = atmo_scatter(m);
            l += t_acc * scat * sun_t * sint;
            f += t_acc * scat * sint;
            t_acc *= exp(-m.ext * dt);
        }
        acc_l += l;
        acc_f += f;
    }
    let f_mean = clamp(acc_f / n, vec3<f32>(0.0), vec3<f32>(0.99));
    let psi = (acc_l / n) / (4.0 * ATMO_PI) / (vec3<f32>(1.0) - f_mean);
    textureStore(multiscatter_out, vec2<i32>(gid.xy), vec4<f32>(psi, 1.0));
}
