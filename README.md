View at https://mzschwartz5.github.io/cesium-interview/ (**on Chrome or browser supporting WebGPU**)

Or run locally:
- Clone repository
- Run `npm install`
- Run `npm run dev`
- Navigate to http://localhost:5173/

Thought process:
- Imagine a simpler problem in 1D: sample heights given along a line. The simplest surface distance estimate would just be a summation of the change in height over each sample point. You could get fancy and fit the data with a higher-order curve, but would that be physical?
- In 2D, the path can and will traverse *between* data points. Simply using a linear estimate between the first and last point would a very rough estimate. More accuracy *requires* some interpolation / fitting a control surface to the data.
- What method of interpolation? How many samples?
  - Treating each grid cell as two triangles, you could send the whole grid to as triangles to the vertex shader, cull the ones the path doesn't pass through, and let the pipeline handle the interpolation (barycentric) for you. However, I'm not sure barycentric interpolation is the best physical match for the underlying data. And it seems wasteful that almost all of the pixels will get culled for not being on the path.
  - Forgetting the triangles: bilinear interpolation would be simple and probably effective. A cubic interpolant may be better because it can guarantee C1 continuity, which terrain tends to be at the scale of the data.
- The process of taking small steps along a path and integrating is highly parallelizable, good for GPU workflows. (Realistically, a CPU is definitely faster for a task like this on a 512x512 image, especially leveraging adaptive step sizes, which would be impractical on the GPU. Buuuuut, I would like to brush off my WebGPU skills, so I ended up attempting a GPU solution).

Difficulties encountered:
- GPU can't handle uint8, which is the format of the height maps. For time's sake, I just passed them as uint32's.
- Initially, my results were the same for pre and post maps - whoops, forgot to copy back the buffer before reusing it.
- Path length not converging with increasing number of samples. Bug with uint underflow in the shader.

Tools used:
- Vite
- Typescript
- WebGPU

Resources used:
- Referenced this repo for some WebGPU skeleton code (a project from UPenn GPU CIS 5650) https://github.com/mzschwartz5/WebGPU-Forward-Plus-and-Clustered-Deferred
- UPenn GPU CIS 5650 slides on parallel reduction: https://docs.google.com/presentation/d/1OHkJLJXptnVFfFG-kTuQsmdQYQjFBRUj/edit#slide=id.p27
- Inigo Quilez for formulation of cubic interpolation: https://youtu.be/BFld4EBO2RE?si=M4FFYC9lvgf1BtVs&t=131
- A little copilot - mostly for speeding up the pipeline set up and some basic HTML. It's usefulness for the kernel code is very limited, often counterproductive.

Time spent: probably a good 6 hours or so... a full evening. I know the prompt called for 2-3 hours - hopefully it's clear that the extra time I needed wasn't due to a lack of ability to solve the problem, but because I chose to write the solution for the GPU.