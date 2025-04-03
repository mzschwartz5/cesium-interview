import './style.css'

async function initializeWebGPU() {
  if (!navigator.gpu) {
    let errorMessage = "WebGPU is not supported on this browser.";
    let errorElement = document.createElement('div');
    errorElement.textContent = errorMessage;
    errorElement.style.color = 'red';

    document.body.appendChild(errorElement);

    throw new Error(errorMessage);
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Failed to get GPU adapter.");
  }

  const device = await adapter.requestDevice();
  if (!device) {
    throw new Error("Failed to get GPU device.");
  }

  return device;
}

async function loadHeightMap(filePath: string): Promise<Uint8Array> {
  const response = await fetch(filePath);
  if (!response.ok) {
    throw new Error(`Failed to load file: ${response.statusText}`);
  }

  const arrayBuffer = await response.arrayBuffer();
  const heightMap = new Uint8Array(arrayBuffer);

  return heightMap;
}

initializeWebGPU();
let preHeightMap = loadHeightMap('/pre.data');
let postHeightMap = loadHeightMap('/post.data');