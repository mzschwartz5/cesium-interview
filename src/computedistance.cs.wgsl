struct Uniforms {
    startX: f32,
    startY: f32,
    endX: f32,
    endY: f32,
    samples: f32,
    mapSizeX: f32,
};

@group(0) @binding(0) var<storage, read> heightMap: array<u32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> distances: array<f32>;


// Groupshared memory
// Somewhat arbitrarily chose the maximum size, though next power of two would exceed WGPU limits (including s_distances).
// For a 512x512 input height map, this means we should be using at least 16 workgroups
// to be absolutely sure we don't exceed this limit in any one workgroup.
var<workgroup> s_heightMap: array<u32, (32 * 32)>;
var<workgroup> s_distances: array<f32, ${workgroupSizeX}>;

fn to1D(x: u32, y: u32, width: u32) -> u32 {
    return x + (y * width);
}

fn to2D(i: u32, width: u32) -> vec2<u32> {
    return vec2<u32>(i % width, i / width);
}

fn cubic(p: f32) -> f32{
    return 3 * p * p - 2 * p * p * p;
}

fn interpolate(
    a: f32, b: f32, c: f32, d: f32, // corners of the cell
    x: f32, y: f32                  // coordinates of the sample point (truncated to the cell)
) -> f32 {
    let cubicX = cubic(x);
    let cubicY = cubic(y);
    return a +
           (b - a) * cubicX +
           (c - a) * cubicY +
           (a - b - c + d) * cubicX * cubicY;
}

@compute @workgroup_size(${workgroupSizeX}, 1, 1)
fn main(
    @builtin(global_invocation_id) gId: vec3<u32>,
    @builtin(local_invocation_id) lId: vec3<u32>,
    @builtin(workgroup_id) wId: vec3<u32>,
) {

    // Vector from the start to end point (in the XY plane of the height map)
    var line = vec2<f32>(
        uniforms.endX - uniforms.startX,
        uniforms.endY - uniforms.startY
    );
    let lineLength = length(line);
    line = (line / lineLength);
    let lineLengthPerWkgrp = lineLength / ceil(uniforms.samples / ${workgroupSizeX});
    let lineLengthPerSample = lineLength / uniforms.samples;

    let start = vec2(uniforms.startX, uniforms.startY);
    let workgroupStart = start + line * lineLengthPerWkgrp * f32(wId.x);
    let workgroupEnd = start + line * lineLengthPerWkgrp * f32(wId.x + 1); // this is conservative for the final workgroup

    // A naive implementation would have a lot of redundant global memory accesses. If we simply preload the m x n cells bounding the line segment operated on by this workgroup,
    // we can reduce most of that redundant access. Divvy up the loading work among the threads in the workgroup.
    let maxBounds = vec2<u32>(ceil(max(workgroupStart, workgroupEnd)));
    let minBounds = vec2<u32>(floor(min(workgroupStart, workgroupEnd)));
    let m = max(maxBounds.x - minBounds.x + 1, 2); // + 1 because vertices = edges + 1 in a grid, max 2 because we need at least 1 edge length
    let n = max(maxBounds.y - minBounds.y + 1, 2); // + 1 because vertices = edges + 1 in a grid, max 2 because we need at least 1 edge length
    let totalPixelsToLoad = m * n;
    let pixelsPerThread = u32(ceil(f32(totalPixelsToLoad) / ${workgroupSizeX}));

    for (var i: u32 = 0; i < pixelsPerThread; i = i + 1) {
        let pixelIndex = (${workgroupSizeX} * i) + lId.x; // indexing this way is best for data access coherency.

        if (pixelIndex >= totalPixelsToLoad) { break;}

        let pixelIndex2D = to2D(pixelIndex, m);
        let globalIndex = to1D(
                minBounds.x + pixelIndex2D.x,
                minBounds.y + pixelIndex2D.y,
                u32(uniforms.mapSizeX)
        );

        if (globalIndex >= u32(uniforms.mapSizeX * uniforms.mapSizeX)) { break; }
        s_heightMap[pixelIndex] = heightMap[globalIndex];
    }
    workgroupBarrier();

    // Now that we have all the data, calculate the distance. Each thread calculates one sample and writes the result to shared memory.
    // WGSL doesn't support early returns when using workgroup barriers, so we have to do this instead.
    if (gId.x < u32(uniforms.samples)) {
        // Determine what cell the sample starts and ends in
        let sampleStartGlobal = start + line * lineLengthPerSample * f32(gId.x);
        let sampleEndGlobal = start + line * lineLengthPerSample * f32(gId.x + 1);
        let sampleCellStartGlobal = vec2<u32>(floor(sampleStartGlobal));
        let sampleCellEndGlobal = vec2<u32>(ceil(sampleEndGlobal - 1f));
        let sampleCellStart = sampleCellStartGlobal - minBounds;
        let sampleCellEnd = sampleCellEndGlobal - minBounds;

        // Get the heightmap values for the 8 pixels (corners of the cells - likely to overlap, but accessing shared memory is basically free anyway).
        // Interpolate the values at each of the 4 corners of the cell to get the height at the sample point
        // We'll use a cubic polynomial surface patch for this - explanation given in write up.
        var a = f32(s_heightMap[to1D(sampleCellStart.x, sampleCellStart.y, m)]);
        var b = f32(s_heightMap[to1D(sampleCellStart.x + 1, sampleCellStart.y, m)]);
        var c = f32(s_heightMap[to1D(sampleCellStart.x, sampleCellStart.y + 1, m)]);
        var d = f32(s_heightMap[to1D(sampleCellStart.x + 1, sampleCellStart.y + 1, m)]);
        let sample1Height = interpolate(a, b, c, d, sampleStartGlobal.x - f32(sampleCellStartGlobal.x), sampleStartGlobal.y - f32(sampleCellStartGlobal.y));

        a = f32(s_heightMap[to1D(sampleCellEnd.x, sampleCellEnd.y, m)]);
        b = f32(s_heightMap[to1D(sampleCellEnd.x + 1, sampleCellEnd.y, m)]);
        c = f32(s_heightMap[to1D(sampleCellEnd.x, sampleCellEnd.y + 1, m)]);
        d = f32(s_heightMap[to1D(sampleCellEnd.x + 1, sampleCellEnd.y + 1, m)]);
        let sample2Height = interpolate(a, b, c, d, sampleEndGlobal.x - f32(sampleCellEndGlobal.x), sampleEndGlobal.y - f32(sampleCellEndGlobal.y));

        // Calculate the distance traveled over the sample
        let sample1To2 = vec3<f32>(
            ${metersPerPixel} * (sampleEndGlobal.x - sampleStartGlobal.x),
            ${metersPerPixel} * (sampleEndGlobal.y - sampleStartGlobal.y),
            ${metersPerHeightValue} * (sample2Height - sample1Height)
        );
        let sample1To2Length = length(sample1To2);

        s_distances[lId.x] = sample1To2Length;
    }


    // Finally, sum the distances in the workgroup and write the result to global memory.
    // We can use parallel reduction to do this efficiently - and we'll just have the CPU sum the results of each workgroup.
    // We're going to enforce the workgroup size to be a power of two to make this simpler.
    // Parallel reduction can be done in place, and if done the right way, allows threads to retire early (or it would... if WGSL supported early returns).
    for (var stride: u32 = ${workgroupSizeX} / 2; stride > 0; stride = stride / 2) {
        workgroupBarrier();
        if (lId.x < stride) {
            s_distances[lId.x] += s_distances[lId.x + stride];
        }
    }

    distances[wId.x] = s_distances[0];
}