import './style.css'
import computeDistanceShader from './computedistance.cs.wgsl?raw';

class Uniforms {
  readonly buffer = new ArrayBuffer(6 * 4); // 6 floats
  private readonly view = new Float32Array(this.buffer);

  set startX(x: number) {
    this.view[0] = x;
  }

  set startY(y: number) {
    this.view[1] = y;
  }

  set endX(x: number) {
    this.view[2] = x;
  }

  set endY(y: number) {
    this.view[3] = y;
  }

  set numSamples(num: number) {
    this.view[4] = num;
  }

  get numSamples() {
    return this.view[4];
  }

  set mapSizeX(size: number) {
    this.view[5] = size;
  }
}

let computeBindGroupLayout: GPUBindGroupLayout;
let preMapBindGroup: GPUBindGroup;
let postMapBindGroup: GPUBindGroup;
let computePipeline: GPUComputePipeline;
let preHeightMap: GPUBuffer;
let postHeightMap: GPUBuffer;
let uniforms: Uniforms = new Uniforms();
let uniformBuffer: GPUBuffer;
let distanceResultsBuffer: GPUBuffer;
let preHeightReadbackBuffer: GPUBuffer;
let postHeightReadbackBuffer: GPUBuffer;
let device: GPUDevice;

const constants = {
  workgroupSizeX: 256,
  metersPerHeightValue: 11,
  metersPerPixel: 30,
};

function evalShaderRaw(raw: string) {
  return eval('`' + raw.replaceAll('${', '${constants.') + '`');
}

async function initializeWebGPU() {
  if (!navigator.gpu) {
    const errorMessage = "WebGPU is not supported on this browser.";
    const errorElement = document.createElement('div');
    errorElement.textContent = errorMessage;
    errorElement.style.color = 'red';

    document.body.appendChild(errorElement);

    throw new Error(errorMessage);
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter.");
  }

  device = await adapter.requestDevice();
  if (!device) {
    throw new Error("Failed to get GPU device.");
  }

  console.log("WebGPU init successsful");

  computeBindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage"},
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
    ],
    label: "computeBindGroupLayout",
  });

  computePipeline = device.createComputePipeline({
    compute: {
      module: device.createShaderModule({
        code: evalShaderRaw(computeDistanceShader),
        label: "computeDistanceShader",
      }),
      entryPoint: "main",
    },
    layout: device.createPipelineLayout({
      bindGroupLayouts: [computeBindGroupLayout],
    }),
  });

  // Load in height maps and copy to GPU buffers
  const preHeightMapPromise = loadHeightMap('/pre.data');
  const postHeightMapPromise = loadHeightMap('/post.data');
  [preHeightMap, postHeightMap] = await Promise.all([preHeightMapPromise, postHeightMapPromise]);

  console.log("Height maps loaded");

  // Create uniform buffer
  uniformBuffer = device.createBuffer({
    label: "uniformBuffer",
    size: uniforms.buffer.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  uniforms.numSamples = 1024;

  // Create distance results buffer
  distanceResultsBuffer = device.createBuffer({
    label: "distanceResultsBuffer",
    size: Math.ceil(uniforms.numSamples / constants.workgroupSizeX) * 4, // 1 float per workgroup
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  preHeightReadbackBuffer = device.createBuffer({
    size: distanceResultsBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  postHeightReadbackBuffer = device.createBuffer({
    size: distanceResultsBuffer.size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  preMapBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: preHeightMap,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: uniformBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: distanceResultsBuffer,
        },
      },
    ],
    label: "preMapBindGroup",
  });

  postMapBindGroup = device.createBindGroup({
    layout: computeBindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: postHeightMap,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: uniformBuffer,
        },
      },
      {
        binding: 2,
        resource: {
          buffer: distanceResultsBuffer,
        },
      },
    ],
    label: "postMapBindGroup",
  });

  console.log("Bind groups created");
}

async function loadHeightMap(filePath: string): Promise<GPUBuffer> {
  const response = await fetch(filePath);
  if (!response.ok) {
    throw new Error(`Failed to load file: ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const uint8Array = new Uint8Array(arrayBuffer);

  // Since GPU doesn't support uint8, we need to convert it to uint32
  // Can optimize later by packing 4 uint8s into a single uint32
  const uint32Array = new Uint32Array(uint8Array.length);
  for (let i = 0; i < uint8Array.length; i++) {
    uint32Array[i] = uint8Array[i];
  }

  const gpuBuf = device.createBuffer({
    size: uint32Array.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
    label: filePath, // just so they're uniquely identified, makes debugging easier
  });

  new Uint32Array(gpuBuf.getMappedRange()).set(uint32Array);
  gpuBuf.unmap();

  return gpuBuf;
}

async function computeDistance(start: { x: number, y: number }, end: { x: number, y: number }): Promise<{ preDistance: number, postDistance: number }> {
  uniforms.startX = start.x;
  uniforms.startY = start.y;
  uniforms.endX = end.x;
  uniforms.endY = end.y;
  uniforms.mapSizeX = 512.0; // could make this more dynamic by taking the sqrt of the height map buffer length

  device.queue.writeBuffer(
    uniformBuffer,
    0,
    uniforms.buffer
  );

  const commandEncoder = device.createCommandEncoder();
  const numWorkGroupsX = Math.ceil(uniforms.numSamples / constants.workgroupSizeX);

  const preComputePass = commandEncoder.beginComputePass();
  preComputePass.setPipeline(computePipeline);

  preComputePass.setBindGroup(0, preMapBindGroup);
  preComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

  preComputePass.end();

  // Copy the results back to the readback buffers
  commandEncoder.copyBufferToBuffer(
    distanceResultsBuffer,
    0,
    preHeightReadbackBuffer,
    0,
    distanceResultsBuffer.size
  );

  const postComputePass = commandEncoder.beginComputePass();
  postComputePass.setPipeline(computePipeline);
  postComputePass.setBindGroup(0, postMapBindGroup);
  postComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

  postComputePass.setBindGroup(0, postMapBindGroup);
  postComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

  postComputePass.end();


  commandEncoder.copyBufferToBuffer(
    distanceResultsBuffer,
    0,
    postHeightReadbackBuffer,
    0,
    distanceResultsBuffer.size
  );

  device.queue.submit([commandEncoder.finish()]);

  // Read back the results
  const preHeightPromise = preHeightReadbackBuffer.mapAsync(GPUMapMode.READ, 0, preHeightReadbackBuffer.size);
  const postHeightPromise = postHeightReadbackBuffer.mapAsync(GPUMapMode.READ, 0, postHeightReadbackBuffer.size);
  await Promise.all([preHeightPromise, postHeightPromise]);
  const preDistancePartialSums = new Float32Array(preHeightReadbackBuffer.getMappedRange());
  const postDistancePartialSums = new Float32Array(postHeightReadbackBuffer.getMappedRange());

  const preDistance = preDistancePartialSums.reduce((acc, val) => acc + val, 0);
  const postDistance = postDistancePartialSums.reduce((acc, val) => acc + val, 0);

  preHeightReadbackBuffer.unmap();
  postHeightReadbackBuffer.unmap();

  return { preDistance, postDistance };
}

await initializeWebGPU();
const { preDistance, postDistance } = await computeDistance({ x: 167, y: 316 }, { x: 317, y: 316 });
console.log(`Pre Distance: ${preDistance} meters, Post Distance: ${postDistance} meters`);
