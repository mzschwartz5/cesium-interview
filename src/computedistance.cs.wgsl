struct Uniforms {
    startX: u32,
    startY: u32,
    endX: u32,
    endY: u32,
    samples: u32,
};

@group(0) @binding(0) var<storage, read> heightMap: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(${workgroupSize})
fn main(@builtin(global_invocation_id) gId: vec3<u32>) {
    let index = gId.x;
    if (index >= uniforms.samples) {
        return;
    }

    let height = heightMap[index];
}