struct Params {
    ns: u32,
    n_z_n_t: u32,
    points: u32,
    force_fields: u32,
    output_fields: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
};
struct Values { data: array<f32>, };
// Ten axisymmetric or sixteen 3-D MHD force fields.
@group(0) @binding(0) var<storage, read> force: Values;
@group(0) @binding(1) var<storage, read> geometry: Values;
// rCon, zCon, rCon0, zCon0, gCon.
@group(0) @binding(2) var<storage, read> constraint: Values;
@group(0) @binding(3) var<storage, read> sqrt_s_f: Values;
// Fourteen axisymmetric or twenty 3-D forward-transform input fields.
@group(0) @binding(4) var<storage, read_write> output: Values;
@group(0) @binding(5) var<uniform> params: Params;

fn con(field: u32, point: u32) -> f32 {
    return constraint.data[field * params.points + point];
}
fn geom(field: u32, point: u32) -> f32 {
    return geometry.data[field * params.points + point];
}
fn put(field: u32, point: u32, value: f32) {
    output.data[field * params.points + point] = value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let point = invocation.x;
    if (point >= params.points) { return; }
    for (var field = 0u; field < params.force_fields; field++) {
        put(field, point, force.data[field * params.points + point]);
    }
    let surface = point / params.n_z_n_t;
    let constraint_offset = params.output_fields - 4u;
    if (surface == 0u) {
        for (var field = constraint_offset; field < params.output_fields;
             field++) {
            put(field, point, 0.0);
        }
        return;
    }
    let sqrt_s = sqrt_s_f.data[surface];
    let dr = con(0u, point) - con(2u, point);
    let dz = con(1u, point) - con(3u, point);
    let gc = con(4u, point);
    let brcon = dr * gc;
    let bzcon = dz * gc;
    put(4u, point, force.data[4u * params.points + point] + brcon);
    put(5u, point, force.data[5u * params.points + point] + brcon * sqrt_s);
    put(6u, point, force.data[6u * params.points + point] + bzcon);
    put(7u, point, force.data[7u * params.points + point] + bzcon * sqrt_s);
    let ru = geom(3u, point) + sqrt_s * geom(9u, point);
    let zu = geom(4u, point) + sqrt_s * geom(10u, point);
    put(constraint_offset + 0u, point, ru * gc);
    put(constraint_offset + 1u, point, ru * gc * sqrt_s);
    put(constraint_offset + 2u, point, zu * gc);
    put(constraint_offset + 3u, point, zu * gc * sqrt_s);
}
