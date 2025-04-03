import './style.css'
import computeDistanceShader from './computedistance.cs.wgsl?raw';

class Uniforms {
  readonly buffer = new ArrayBuffer(5 * 4);
  private readonly view = new Uint32Array(this.buffer);

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
}

let computeBindGroupLayout: GPUBindGroupLayout;
let preMapBindGroup: GPUBindGroup;
let postMapBindGroup: GPUBindGroup;
let computePipeline: GPUComputePipeline;
let preHeightMap: GPUBuffer;
let postHeightMap: GPUBuffer;
let uniforms: Uniforms = new Uniforms();
let uniformBuffer: GPUBuffer;
let device: GPUDevice;

const constants = {
  workgroupSize: [32, 1, 1],
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
      }
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

async function computeDistance(start: {x: number, y: number}, end: {x: number, y: number}) {
  uniforms.startX = start.x;
  uniforms.startY = start.y;
  uniforms.endX = end.x;
  uniforms.endY = end.y;
  uniforms.numSamples = 100;

  device.queue
        .writeBuffer(
          uniformBuffer, 0,
          uniforms.buffer
        );

  const commandEncoder = device.createCommandEncoder();
  const computePass = commandEncoder.beginComputePass();
  computePass.setPipeline(computePipeline);

  const workGroupSizeX = Math.ceil(uniforms.numSamples / constants.workgroupSize[0]);
  computePass.setBindGroup(0, preMapBindGroup);
  computePass.dispatchWorkgroups(workGroupSizeX, 1, 1);

  computePass.setBindGroup(0, postMapBindGroup);
  computePass.dispatchWorkgroups(workGroupSizeX, 1, 1);

  computePass.end();
  device.queue.submit([commandEncoder.finish()]);
}

await initializeWebGPU();
computeDistance({ x: 0, y: 0 }, { x: 512, y: 512 })
