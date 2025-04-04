var m=Object.defineProperty;var g=(t,e,a)=>e in t?m(t,e,{enumerable:!0,configurable:!0,writable:!0,value:a}):t[e]=a;var s=(t,e,a)=>g(t,typeof e!="symbol"?e+"":e,a);(function(){const e=document.createElement("link").relList;if(e&&e.supports&&e.supports("modulepreload"))return;for(const n of document.querySelectorAll('link[rel="modulepreload"]'))i(n);new MutationObserver(n=>{for(const r of n)if(r.type==="childList")for(const o of r.addedNodes)o.tagName==="LINK"&&o.rel==="modulepreload"&&i(o)}).observe(document,{childList:!0,subtree:!0});function a(n){const r={};return n.integrity&&(r.integrity=n.integrity),n.referrerPolicy&&(r.referrerPolicy=n.referrerPolicy),n.crossOrigin==="use-credentials"?r.credentials="include":n.crossOrigin==="anonymous"?r.credentials="omit":r.credentials="same-origin",r}function i(n){if(n.ep)return;n.ep=!0;const r=a(n);fetch(n.href,r)}})();const computeDistanceShader=`struct Uniforms {
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
var<workgroup> s_distances: array<f32, \${workgroupSizeX}>;

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

@compute @workgroup_size(\${workgroupSizeX}, 1, 1)
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
    let lineLengthPerWkgrp = lineLength / ceil(uniforms.samples / \${workgroupSizeX});
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
    let pixelsPerThread = u32(ceil(f32(totalPixelsToLoad) / \${workgroupSizeX}));

    for (var i: u32 = 0; i < pixelsPerThread; i = i + 1) {
        let pixelIndex = (\${workgroupSizeX} * i) + lId.x; // indexing this way is best for data access coherency.

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
            \${metersPerPixel} * (sampleEndGlobal.x - sampleStartGlobal.x),
            \${metersPerPixel} * (sampleEndGlobal.y - sampleStartGlobal.y),
            \${metersPerHeightValue} * (sample2Height - sample1Height)
        );
        let sample1To2Length = length(sample1To2);

        s_distances[lId.x] = sample1To2Length;
    }


    // Finally, sum the distances in the workgroup and write the result to global memory.
    // We can use parallel reduction to do this efficiently - and we'll just have the CPU sum the results of each workgroup.
    // We're going to enforce the workgroup size to be a power of two to make this simpler.
    // Parallel reduction can be done in place, and if done the right way, allows threads to retire early (or it would... if WGSL supported early returns).
    for (var stride: u32 = \${workgroupSizeX} / 2; stride > 0; stride = stride / 2) {
        workgroupBarrier();
        if (lId.x < stride) {
            s_distances[lId.x] += s_distances[lId.x + stride];
        }
    }

    distances[wId.x] = s_distances[0];
}`;class Uniforms{constructor(){s(this,"buffer",new ArrayBuffer(6*4));s(this,"view",new Float32Array(this.buffer))}set startX(e){this.view[0]=e}set startY(e){this.view[1]=e}set endX(e){this.view[2]=e}set endY(e){this.view[3]=e}set numSamples(e){this.view[4]=e}get numSamples(){return this.view[4]}set mapSizeX(e){this.view[5]=e}}class Solver{constructor(t){s(this,"computeBindGroupLayout");s(this,"preMapBindGroup");s(this,"postMapBindGroup");s(this,"computePipeline");s(this,"preHeightMap");s(this,"postHeightMap");s(this,"uniformBuffer");s(this,"distanceResultsBuffer");s(this,"preHeightReadbackBuffer");s(this,"postHeightReadbackBuffer");s(this,"device");s(this,"uniforms",new Uniforms);s(this,"constants",{workgroupSizeX:256,metersPerHeightValue:11,metersPerPixel:30});this.device=t}async createGPUResources(){this.createComputePipeline(),await this.createBuffers(),this.createBindGroups()}createComputePipeline(){this.computeBindGroupLayout=this.device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:"read-only-storage"}},{binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:"uniform"}},{binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:"storage"}}],label:"computeBindGroupLayout"}),this.computePipeline=this.device.createComputePipeline({compute:{module:this.device.createShaderModule({code:this.evalShaderRaw(computeDistanceShader),label:"computeDistanceShader"}),entryPoint:"main"},layout:this.device.createPipelineLayout({bindGroupLayouts:[this.computeBindGroupLayout]})}),console.log("Compute pipeline created")}async createBuffers(){const t=this.loadHeightMap("./pre.data"),e=this.loadHeightMap("./post.data");[this.preHeightMap,this.postHeightMap]=await Promise.all([t,e]),console.log("Height maps loaded"),this.uniformBuffer=this.device.createBuffer({label:"uniformBuffer",size:this.uniforms.buffer.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}),this.uniforms.numSamples=1024*8,this.distanceResultsBuffer=this.device.createBuffer({label:"distanceResultsBuffer",size:Math.ceil(this.uniforms.numSamples/this.constants.workgroupSizeX)*4,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),this.preHeightReadbackBuffer=this.device.createBuffer({size:this.distanceResultsBuffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),this.postHeightReadbackBuffer=this.device.createBuffer({size:this.distanceResultsBuffer.size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ}),console.log("Buffers created")}createBindGroups(){this.preMapBindGroup=this.device.createBindGroup({layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.preHeightMap}},{binding:1,resource:{buffer:this.uniformBuffer}},{binding:2,resource:{buffer:this.distanceResultsBuffer}}],label:"preMapBindGroup"}),this.postMapBindGroup=this.device.createBindGroup({layout:this.computeBindGroupLayout,entries:[{binding:0,resource:{buffer:this.postHeightMap}},{binding:1,resource:{buffer:this.uniformBuffer}},{binding:2,resource:{buffer:this.distanceResultsBuffer}}],label:"postMapBindGroup"}),console.log("Bind groups created")}async loadHeightMap(t){const e=await fetch(t);if(!e.ok)throw new Error(`Failed to load file: ${e.statusText}`);const a=await e.arrayBuffer(),i=new Uint8Array(a),n=new Uint32Array(i.length);for(let o=0;o<i.length;o++)n[o]=i[o];const r=this.device.createBuffer({size:n.byteLength,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST,mappedAtCreation:!0,label:t});return new Uint32Array(r.getMappedRange()).set(n),r.unmap(),r}async computeDistance(t,e){this.uniforms.startX=t.x,this.uniforms.startY=t.y,this.uniforms.endX=e.x,this.uniforms.endY=e.y,this.uniforms.mapSizeX=512,this.device.queue.writeBuffer(this.uniformBuffer,0,this.uniforms.buffer);const a=this.device.createCommandEncoder(),i=Math.ceil(this.uniforms.numSamples/this.constants.workgroupSizeX),n=a.beginComputePass();n.setPipeline(this.computePipeline),n.setBindGroup(0,this.preMapBindGroup),n.dispatchWorkgroups(i,1,1),n.end(),a.copyBufferToBuffer(this.distanceResultsBuffer,0,this.preHeightReadbackBuffer,0,this.distanceResultsBuffer.size);const r=a.beginComputePass();r.setPipeline(this.computePipeline),r.setBindGroup(0,this.postMapBindGroup),r.dispatchWorkgroups(i,1,1),r.setBindGroup(0,this.postMapBindGroup),r.dispatchWorkgroups(i,1,1),r.end(),a.copyBufferToBuffer(this.distanceResultsBuffer,0,this.postHeightReadbackBuffer,0,this.distanceResultsBuffer.size),this.device.queue.submit([a.finish()]);const o=this.preHeightReadbackBuffer.mapAsync(GPUMapMode.READ,0,this.preHeightReadbackBuffer.size),p=this.postHeightReadbackBuffer.mapAsync(GPUMapMode.READ,0,this.postHeightReadbackBuffer.size);await Promise.all([o,p]);const d=new Float32Array(this.preHeightReadbackBuffer.getMappedRange()),c=new Float32Array(this.postHeightReadbackBuffer.getMappedRange()),f=d.reduce((l,u)=>l+u,0),h=c.reduce((l,u)=>l+u,0);return this.preHeightReadbackBuffer.unmap(),this.postHeightReadbackBuffer.unmap(),{preDistance:f,postDistance:h}}evalShaderRaw(raw){return eval("`"+raw.replaceAll("${","${this.constants.")+"`")}}async function initializeWebGPU(){if(!navigator.gpu){const a="WebGPU is not supported on this browser.",i=document.createElement("div");throw i.textContent=a,i.style.color="red",document.body.appendChild(i),new Error(a)}const t=await navigator.gpu.requestAdapter();if(!t)throw new Error("Failed to get GPU adapter.");const e=await t.requestDevice();if(!e)throw new Error("Failed to get GPU device.");return console.log("WebGPU init successsful"),e}async function main(){let t=await initializeWebGPU(),e=new Solver(t);await e.createGPUResources();const a=document.getElementById("startX"),i=document.getElementById("startY"),n=document.getElementById("endX"),r=document.getElementById("endY"),o=document.getElementById("computeButton"),p=document.getElementById("result");o.addEventListener("click",async()=>{const d=parseFloat(a.value),c=parseFloat(i.value),f=parseFloat(n.value),h=parseFloat(r.value);if(!validateInput(d)||!validateInput(c)||!validateInput(f)||!validateInput(h)){p.textContent="Please enter a number between 0-511 for all fields.";return}const{preDistance:l,postDistance:u}=await e.computeDistance({x:d,y:c},{x:f,y:h});p.textContent=`Difference between pre-eruption and post-eruption distances: ${Math.abs(l-u).toFixed(2)}`})}function validateInput(t){return!(isNaN(t)||t<0||t>512)}main();
