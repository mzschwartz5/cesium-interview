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

export class Solver {
    computeBindGroupLayout: GPUBindGroupLayout;
    preMapBindGroup: GPUBindGroup;
    postMapBindGroup: GPUBindGroup;
    computePipeline: GPUComputePipeline;
    preHeightMap: GPUBuffer;
    postHeightMap: GPUBuffer;
    uniformBuffer: GPUBuffer;
    distanceResultsBuffer: GPUBuffer;
    preHeightReadbackBuffer: GPUBuffer;
    postHeightReadbackBuffer: GPUBuffer;
    device: GPUDevice;
    uniforms: Uniforms = new Uniforms();

    constants = {
        workgroupSizeX: 256,
        metersPerHeightValue: 11,
        metersPerPixel: 30,
    };

    constructor(device: GPUDevice) {
        this.device = device;
    }

    async createGPUResources() {
        this.createComputePipeline();
        await this.createBuffers();
        this.createBindGroups();
    }

    createComputePipeline() {
        this.computeBindGroupLayout = this.device.createBindGroupLayout({
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

        this.computePipeline = this.device.createComputePipeline({
            compute: {
              module: this.device.createShaderModule({
                code: this.evalShaderRaw(computeDistanceShader),
                label: "computeDistanceShader",
              }),
              entryPoint: "main",
            },
            layout: this.device.createPipelineLayout({
              bindGroupLayouts: [this.computeBindGroupLayout],
            }),
        });

        console.log("Compute pipeline created");
    }

    async createBuffers() {
        // Load in height maps and copy to GPU buffers
        const preHeightMapPromise = this.loadHeightMap('/pre.data');
        const postHeightMapPromise = this.loadHeightMap('/post.data');
        [this.preHeightMap, this.postHeightMap] = await Promise.all([preHeightMapPromise, postHeightMapPromise]);

        console.log("Height maps loaded");

        // Create uniform buffer
        this.uniformBuffer = this.device.createBuffer({
            label: "uniformBuffer",
            size: this.uniforms.buffer.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.uniforms.numSamples = 1024; // use this to control the integration resolution

        // Create distance results buffer
        this.distanceResultsBuffer = this.device.createBuffer({
            label: "distanceResultsBuffer",
            size: Math.ceil(this.uniforms.numSamples / this.constants.workgroupSizeX) * 4, // 1 float per workgroup
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });

        this.preHeightReadbackBuffer = this.device.createBuffer({
            size: this.distanceResultsBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        this.postHeightReadbackBuffer = this.device.createBuffer({
            size: this.distanceResultsBuffer.size,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

        console.log("Buffers created");
    }

    createBindGroups() {
        this.preMapBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
              {
                binding: 0,
                resource: {
                  buffer: this.preHeightMap,
                },
              },
              {
                binding: 1,
                resource: {
                  buffer: this.uniformBuffer,
                },
              },
              {
                binding: 2,
                resource: {
                  buffer: this.distanceResultsBuffer,
                },
              },
            ],
            label: "preMapBindGroup",
        });

        this.postMapBindGroup = this.device.createBindGroup({
            layout: this.computeBindGroupLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: this.postHeightMap,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: this.uniformBuffer,
                    },
                },
                {
                    binding: 2,
                    resource: {
                        buffer: this.distanceResultsBuffer,
                    },
                },
            ],
            label: "postMapBindGroup",
        });

        console.log("Bind groups created");
    }

    async loadHeightMap(filePath: string): Promise<GPUBuffer> {
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

        const gpuBuf = this.device.createBuffer({
          size: uint32Array.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          mappedAtCreation: true,
          label: filePath, // just so they're uniquely identified, makes debugging easier
        });

        new Uint32Array(gpuBuf.getMappedRange()).set(uint32Array);
        gpuBuf.unmap();

        return gpuBuf;
    }

    /**
     * This is the actual work horse function that dispatches the compute shader for each height map to compute the topological distance between the start and end pixel.
     *
     * @param start - the start point of the distance calculation (pixel coordinates of height map)
     * @param end - the end point of the distance calculation (pixel coordinates of height map)
     * @returns - two distances, from start point to end point, one for the pre-eruption height map, one for the post-eruption height map
     */
    async computeDistance(start: { x: number, y: number }, end: { x: number, y: number }): Promise<{ preDistance: number, postDistance: number }> {
        this.uniforms.startX = start.x;
        this.uniforms.startY = start.y;
        this.uniforms.endX = end.x;
        this.uniforms.endY = end.y;
        this.uniforms.mapSizeX = 512.0; // could make this more dynamic by taking the sqrt of the height map buffer length

        this.device.queue.writeBuffer(
          this.uniformBuffer,
          0,
          this.uniforms.buffer
        );

        const commandEncoder = this.device.createCommandEncoder();
        const numWorkGroupsX = Math.ceil(this.uniforms.numSamples / this.constants.workgroupSizeX);

        const preComputePass = commandEncoder.beginComputePass();
        preComputePass.setPipeline(this.computePipeline);

        preComputePass.setBindGroup(0, this.preMapBindGroup);
        preComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

        preComputePass.end();

        // Copy the results back to the readback buffers
        commandEncoder.copyBufferToBuffer(
          this.distanceResultsBuffer,
          0,
          this.preHeightReadbackBuffer,
          0,
          this.distanceResultsBuffer.size
        );

        const postComputePass = commandEncoder.beginComputePass();
        postComputePass.setPipeline(this.computePipeline);
        postComputePass.setBindGroup(0, this.postMapBindGroup);
        postComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

        postComputePass.setBindGroup(0, this.postMapBindGroup);
        postComputePass.dispatchWorkgroups(numWorkGroupsX, 1, 1);

        postComputePass.end();

        commandEncoder.copyBufferToBuffer(
          this.distanceResultsBuffer,
          0,
          this.postHeightReadbackBuffer,
          0,
          this.distanceResultsBuffer.size
        );

        this.device.queue.submit([commandEncoder.finish()]);

        // Read back the results
        const preHeightPromise = this.preHeightReadbackBuffer.mapAsync(GPUMapMode.READ, 0, this.preHeightReadbackBuffer.size);
        const postHeightPromise = this.postHeightReadbackBuffer.mapAsync(GPUMapMode.READ, 0, this.postHeightReadbackBuffer.size);
        await Promise.all([preHeightPromise, postHeightPromise]);
        const preDistancePartialSums = new Float32Array(this.preHeightReadbackBuffer.getMappedRange());
        const postDistancePartialSums = new Float32Array(this.postHeightReadbackBuffer.getMappedRange());

        const preDistance = preDistancePartialSums.reduce((acc, val) => acc + val, 0);
        const postDistance = postDistancePartialSums.reduce((acc, val) => acc + val, 0);

        this.preHeightReadbackBuffer.unmap();
        this.postHeightReadbackBuffer.unmap();

        return { preDistance, postDistance };
    }

    /**
     * This is just a helper to replace ${} values in the .wgsl file with constant values
     *
     * @param raw - the raw shader code
     * @returns the shader with the constants replaced with their values
     */
    evalShaderRaw(raw: string) {
        return eval('`' + raw.replaceAll('${', '${this.constants.') + '`');
    }
}