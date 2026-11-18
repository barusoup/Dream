struct ReflectionParams {
    size: vec4<f32>, full: vec4<f32>, depth: vec4<f32>, proj: vec4<f32>,
    view_from_world: mat4x4<f32>, trace: vec4<f32>, surface: vec4<f32>,
    frame: vec4<f32>, temporal: vec4<f32>, world_from_view: mat4x4<f32>,
    sun_direction: vec4<f32>, sun_color: vec4<f32>, sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>, sky_ground: vec4<f32>, env_valid: vec4<f32>,
};
struct RtMaterial { base: vec4<f32>, emissive: vec4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> P: ReflectionParams;
@group(0) @binding(1) var linear_depth: texture_2d<f32>;
@group(0) @binding(2) var surface_tex: texture_2d<f32>;
@group(0) @binding(3) var colour_tex: texture_2d<f32>;
@group(0) @binding(4) var out_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var scene: acceleration_structure<vertex_return>;
@group(0) @binding(6) var<storage, read> materials: array<RtMaterial>;
@group(0) @binding(7) var environment: texture_2d<f32>;
@group(0) @binding(8) var env_sampler: sampler;
const PI: f32 = 3.14159265359;
const SKY_DEPTH: f32 = 65000.0;
const VNDF_LO: f32 = 0.075;
const VNDF_HI: f32 = 0.925;
fn decode_normal(s: vec4<f32>) -> vec3<f32> {
    let e = s.xy * 2.0 - vec2<f32>(1.0);
    var o = e;
    let folded = abs(e.x) + abs(e.y) > 1.0;
    if folded { o = (vec2<f32>(1.0) - abs(e.yx)) * vec2<f32>(select(-1.0, 1.0, e.x >= 0.0), select(-1.0, 1.0, e.y >= 0.0)); }
    return normalize(vec3<f32>(o.x, (1.0 - abs(o.x) - abs(o.y)) * select(1.0, -1.0, folded), o.y));
}
fn environment_uv(d: vec3<f32>) -> vec2<f32> {
    return vec2<f32>(atan2(d.z, d.x) / (2.0 * PI) + 0.5, acos(clamp(d.y, -1.0, 1.0)) / PI);
}
fn fallback_sky(d: vec3<f32>) -> vec3<f32> {
    if P.env_valid.x > 0.5 { return textureSampleLevel(environment, env_sampler, environment_uv(d), 0.0).rgb; }
    if d.y >= 0.0 { return mix(P.sky_horizon.rgb, P.sky_zenith.rgb, pow(clamp(d.y, 0.0, 1.0), 0.65)); }
    return mix(P.sky_horizon.rgb, P.sky_ground.rgb, clamp(-d.y * 1.5, 0.0, 1.0));
}
fn reflection_direction(v: vec3<f32>, n: vec3<f32>, alpha: f32, u1_in: f32, u2_in: f32) -> vec3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z); let b = n.x * n.y * a;
    let tb = vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bb = vec3<f32>(b, s + n.y * n.y * a, -n.y);
    let ve = vec3<f32>(dot(v, tb), dot(v, bb), dot(v, n));
    let vh = normalize(vec3<f32>(alpha * ve.x, alpha * ve.y, ve.z));
    let lensq = vh.x * vh.x + vh.y * vh.y;
    var t1 = vec3<f32>(1.0, 0.0, 0.0);
    if lensq > 0.0 { t1 = vec3<f32>(-vh.y, vh.x, 0.0) / sqrt(lensq); }
    let t2 = cross(vh, t1);
    let u1 = clamp(u1_in, VNDF_LO, VNDF_HI); let u2 = clamp(u2_in, VNDF_LO, VNDF_HI);
    let r = sqrt(u1); let phi = 2.0 * PI * u2; let p1 = r * cos(phi);
    var p2 = r * sin(phi); let hs = 0.5 * (1.0 + vh.z);
    p2 = (1.0 - hs) * sqrt(1.0 - p1 * p1) + hs * p2;
    let p3 = sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2));
    let nh = p1 * t1 + p2 * t2 + p3 * vh;
    let he = normalize(vec3<f32>(alpha * nh.x, alpha * nh.y, max(0.0, nh.z)));
    let h = he.x * tb + he.y * bb + he.z * n;
    var dir = reflect(-v, h);
    if dot(dir, n) <= 0.0 { dir = reflect(-v, n); }
    return normalize(dir);
}
fn visibility(origin: vec3<f32>, direction: vec3<f32>) -> f32 {
    var q: ray_query<vertex_return>;
    rayQueryInitialize(&q, scene, RayDesc(RAY_FLAG_FORCE_OPAQUE | RAY_FLAG_TERMINATE_ON_FIRST_HIT, 255u, 0.025, 10000.0, origin, direction));
    while rayQueryProceed(&q) {}
    let h = rayQueryGetCommittedIntersection(&q);
    return select(1.0, 0.0, h.kind != RAY_QUERY_INTERSECTION_NONE);
}
fn shade_hit(origin: vec3<f32>, direction: vec3<f32>, max_t: f32, pixel: vec2<i32>) -> vec4<f32> {
    var q: ray_query<vertex_return>;
    rayQueryInitialize(&q, scene, RayDesc(RAY_FLAG_FORCE_OPAQUE, 255u, 0.025, max_t, origin, direction));
    while rayQueryProceed(&q) {}
    let hit = rayQueryGetCommittedIntersection(&q);
    if hit.kind == RAY_QUERY_INTERSECTION_NONE { return vec4<f32>(fallback_sky(direction), 1.0); }
    let world_pos = origin + direction * hit.t;
    let view_pos = (P.view_from_world * vec4<f32>(world_pos, 1.0)).xyz;
    if view_pos.z < -P.proj.x {
        let ndc = vec2<f32>(view_pos.x / (-view_pos.z * P.proj.z * P.proj.w), view_pos.y / (-view_pos.z * P.proj.z));
        let uv = ndc * 0.5 + vec2<f32>(0.5);
        if all(uv >= vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) {
            let hp = clamp(vec2<i32>(uv * P.full.xy), vec2<i32>(0), vec2<i32>(P.full.xy) - vec2<i32>(1));
            let dp = clamp(vec2<i32>(uv * P.depth.xy), vec2<i32>(0), vec2<i32>(P.depth.xy) - vec2<i32>(1));
            let hd = textureLoad(linear_depth, dp, 0).r;
            let expected = -view_pos.z;
            if hd < SKY_DEPTH && abs(hd - expected) <= max(0.05, expected * 0.02) {
                return vec4<f32>(textureLoad(colour_tex, hp, 0).rgb, 1.0);
            }
        }
    }
    let verts = getCommittedHitVertexPositions(&q);
    let a = (hit.object_to_world * vec4<f32>(verts[0].xyz, 1.0)).xyz;
    let b = (hit.object_to_world * vec4<f32>(verts[1].xyz, 1.0)).xyz;
    let c = (hit.object_to_world * vec4<f32>(verts[2].xyz, 1.0)).xyz;
    var n = normalize(cross(b - a, c - a));
    if !hit.front_face { n = -n; }
    if dot(n, direction) > 0.0 { n = -n; }
    let mat = materials[hit.instance_custom_data];
    let sun_dir = normalize(P.sun_direction.xyz);
    let vis = visibility(world_pos + n * 0.03, sun_dir);
    let diffuse = max(dot(n, sun_dir), 0.0) * vis;
    let reflectance = 1.0 - clamp(mat.params.y, 0.0, 1.0);
    return vec4<f32>(mat.base.rgb * (fallback_sky(n) * reflectance + P.sun_color.rgb * (diffuse / PI) * reflectance) + mat.emissive.rgb, 1.0);
}
fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
@compute @workgroup_size(8, 8)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = vec2<i32>(P.size.xy); let q = vec2<i32>(gid.xy);
    if any(q >= size) { return; }
    let pixel = clamp(vec2<i32>((vec2<f32>(gid.xy) + vec2<f32>(0.5)) * P.full.xy / P.size.xy), vec2<i32>(0), vec2<i32>(P.full.xy) - vec2<i32>(1));
    let dp = clamp(vec2<i32>((vec2<f32>(gid.xy) + vec2<f32>(0.5)) * P.depth.xy / P.size.xy), vec2<i32>(0), vec2<i32>(P.depth.xy) - vec2<i32>(1));
    let d = textureLoad(linear_depth, dp, 0).r;
    let surf = textureLoad(surface_tex, pixel, 0);
    if d > 64000.0 || surf.a <= 0.1 || surf.b >= P.surface.x { textureStore(out_tex, q, vec4<f32>(0.0)); return; }
    let ndc = (vec2<f32>(pixel) + vec2<f32>(0.5)) / P.full.xy * 2.0 - vec2<f32>(1.0);
    let view_pos = vec3<f32>(ndc.x * P.proj.z * P.proj.w * d, ndc.y * P.proj.z * d, -d);
    let world_pos = (P.world_from_view * vec4<f32>(view_pos, 1.0)).xyz;
    let n = decode_normal(surf);
    let view_world = normalize((P.world_from_view * vec4<f32>(normalize(-view_pos), 0.0)).xyz);
    let alpha = surf.b * surf.b;
    var dir = normalize(reflect(-view_world, n));
    if alpha >= 0.005 {
        let r1 = hash(vec2<f32>(pixel) + P.frame.x); let r2 = hash(vec2<f32>(pixel) + vec2<f32>(17.0, 9.0) + P.frame.x);
        dir = reflection_direction(view_world, n, alpha, r1, r2);
    }
    var color = fallback_sky(dir);
    let hit = shade_hit(world_pos + n * 0.025, dir, P.trace.x, pixel);
    color = hit.rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    if luminance > 8.0 { color *= 8.0 / luminance; }
    textureStore(out_tex, q, vec4<f32>(color, 1.0));
}
