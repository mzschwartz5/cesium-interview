export async function initializeWebGPU(): Promise<GPUDevice> {
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
  
    const device = await adapter.requestDevice();
    if (!device) {
      throw new Error("Failed to get GPU device.");
    }

    console.log("WebGPU init successsful");
    return device;
}