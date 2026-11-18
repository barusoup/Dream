struct IndirectParams {
    size: vec4<f32>, full: vec4<f32>, proj: vec4<f32>, view_from_world: mat4x4<f32>,
    ao: vec4<f32>, gi: vec4<f32>, march: vec4<f32>, temporal: vec4<f32>,
    world_from_view: mat4x4<f32>, sun_direction: vec4<f32>, sun_color: vec4<f32>,
    sky_zenith: vec4<f32>, sky_horizon: vec4<f32>, sky_ground: vec4<f32>, env_valid: vec4<f32>,
};
struct RtMaterial { base: vec4<f32>, emissive: vec4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> P: IndirectParams;
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
@group(0) @binding(2) var surface_tex: texture_2d<f32>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var scene: acceleration_structure<vertex_return>;
@group(0) @binding(5) var<storage, read> materials: array<RtMaterial>;
@group(0) @binding(6) var environment: texture_2d<f32>;
@group(0) @binding(7) var env_sampler: sampler;
const PI: f32 = 3.14159265359;
const SKY_DEPTH: f32 = 65000.0;
fn decode_normal(s: vec4<f32>) -> vec3<f32> {
    let e = s.xy * 2.0 - vec2<f32>(1.0); var o = e;
    let folded = abs(e.x) + abs(e.y) > 1.0;
    if folded { o = (vec2<f32>(1.0) - abs(e.yx)) * vec2<f32>(select(-1.0, 1.0, e.x >= 0.0), select(-1.0, 1.0, e.y >= 0.0)); }
    return normalize(vec3<f32>(o.x, (1.0 - abs(o.x) - abs(o.y)) * select(1.0, -1.0, folded), o.y));
}
fn env_uv(d: vec3<f32>) -> vec2<f32> { return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI); }
fn sky(d: vec3<f32>) -> vec3<f32> {
    if P.env_valid.x > 0.5 { return textureSampleLevel(environment, env_sampler, env_uv(d), 0.0).rgb; }
    if d.y >= 0.0 { return mix(P.sky_horizon.rgb, P.sky_zenith.rgb, pow(clamp(d.y, 0.0, 1.0), 0.65)); }
    return mix(P.sky_horizon.rgb, P.sky_ground.rgb, clamp(-d.y * 1.5, 0.0, 1.0));
}
fn random2(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453), fract(sin(dot(p, vec2<f32>(269.5, 183.3))) * 43758.5453));
}
fn cosine_dir(n: vec3<f32>, r: vec2<f32>) -> vec3<f32> {
    let radius = sqrt(r.x); let phi = 2.0 * PI * r.y;
    let t = normalize(cross(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.y) > 0.95), n));
    let b = cross(n, t);
    return normalize(t * (radius * cos(phi)) + b * (radius * sin(phi)) + n * sqrt(max(0.0, 1.0 - r.x)));
}
fn occluded(origin: vec3<f32>, direction: vec3<f32>, max_t: f32) -> bool {
    var q: ray_query<vertex_return>;
    rayQueryInitialize(&q, scene, RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, 255u, 0.02, max_t, origin, direction));
    while rayQueryProceed(&q) {}
    return rayQueryGetCommittedIntersection(&q).kind != RAY_QUERY_INTERSECTION_NONE;
}
@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let q = vec2<i32>(gid.xy); if any(q >= vec2<i32>(P.size.xy)) { return; }
    let pixel = clamp(vec2<i32>((vec2<f32>(gid.xy) + vec2<f32>(0.5)) * P.full.xy / P.size.xy), vec2<i32>(0), vec2<i32>(P.full.xy) - vec2<i32>(1));
    let depth_size = vec2<i32>(textureDimensions(linear_depth));
    let dp = clamp(vec2<i32>((vec2<f32>(gid.xy) + vec2<f32>(0.5)) * P.march.zw), vec2<i32>(0), depth_size - vec2<i32>(1));
    let d = textureLoad(linear_depth, dp, 0).r; let surf = textureLoad(surface_tex, pixel, 0);
    if d >= SKY_DEPTH || surf.a <= 0.1 { textureStore(out_tex, q, vec4<f32>(0.0, 0.0, 0.0, 1.0)); return; }
    let ndc = (vec2<f32>(pixel) + vec2<f32>(0.5)) / P.full.xy * 2.0 - vec2<f32>(1.0);
    let view_pos = vec3<f32>(ndc.x * P.proj.z * P.proj.w * d, ndc.y * P.proj.z * d, -d);
    let world_pos = (P.world_from_view * vec4<f32>(view_pos, 1.0)).xyz;
    let n = decode_normal(surf);
    let viewdir = normalize((P.world_from_view * vec4<f32>(normalize(-view_pos), 0.0)).xyz);
    let rnd = random2(vec2<f32>(pixel) + P.march.x * vec2<f32>(17.0, 31.0));
    let direction = cosine_dir(n, rnd);
    let max_t = max(P.ao.x, P.gi.x);
    var query: ray_query<vertex_return>;
    rayQueryInitialize(&query, scene, RayDesc(RAY_FLAG_FORCE_OPAQUE, 255u, 0.02, max_t, world_pos + n * 0.025, direction));
    while rayQueryProceed(&query) {}
    let hit = rayQueryGetCommittedIntersection(&query);
    var visibility = 1.0;
    var radiance = vec3<f32>(0.0);
    if hit.kind != RAY_QUERY_INTERSECTION_NONE {
        if hit.t < P.ao.x { visibility = min(1.0, (hit.t / max(P.ao.x, 1e-3)) * (hit.t / max(P.ao.x, 1e-3))); }
        if hit.t < P.gi.x {
            let verts = getCommittedHitVertexPositions(&query);
            let a = (hit.object_to_world * vec4<f32>(verts[0].xyz, 1.0)).xyz;
            let b = (hit.object_to_world * vec4<f32>(verts[1].xyz, 1.0)).xyz;
            let c = (hit.object_to_world * vec4<f32>(verts[2].xyz, 1.0)).xyz;
            var hn = normalize(cross(b - a, c - a)); if !hit.front_face { hn = -hn; }
            if dot(hn, direction) > 0.0 { hn = -hn; }
            let m = materials[hit.instance_custom_data];
            let sun = normalize(P.sun_direction.xyz);
            let shadow = select(1.0, 0.0, occluded(world_pos + direction * hit.t + hn * 0.03, sun, 10000.0));
            let reflectance = 1.0 - clamp(m.params.y, 0.0, 1.0);
            let sky_irradiance = sky(hn);
            radiance = m.base.rgb * (sky_irradiance * reflectance + P.sun_color.rgb * max(dot(hn, sun), 0.0) * shadow * reflectance / PI) + m.emissive.rgb;
        } else { radiance = vec3<f32>(0.0); }
    }
    if P.ao.z < 0.5 { visibility = 1.0; }
    if P.gi.y < 0.5 { radiance = vec3<f32>(0.0); }
    let luminance = dot(radiance, vec3<f32>(0.2126, 0.7152, 0.0722));
    if luminance > 8.0 { radiance *= 8.0 / luminance; }
    textureStore(out_tex, q, vec4<f32>(radiance, visibility));
}
