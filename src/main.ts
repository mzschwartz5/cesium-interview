import './style.css'
import { Solver } from './solver';
import { initializeWebGPU } from './webgpu';

async function main() {
  let device = await initializeWebGPU();
  let solver = new Solver(device);
  await solver.createGPUResources();

  // Basic HTML elements for user input
  const startXInput = document.getElementById('startX') as HTMLInputElement;
  const startYInput = document.getElementById('startY') as HTMLInputElement;
  const endXInput = document.getElementById('endX') as HTMLInputElement;
  const endYInput = document.getElementById('endY') as HTMLInputElement;
  const computeButton = document.getElementById('computeButton') as HTMLButtonElement;
  const resultDisplay = document.getElementById('result') as HTMLDivElement;

  computeButton.addEventListener('click', async () => {
    const startX = parseFloat(startXInput.value);
    const startY = parseFloat(startYInput.value);
    const endX = parseFloat(endXInput.value);
    const endY = parseFloat(endYInput.value);

    if (!validateInput(startX) || !validateInput(startY) || !validateInput(endX) || !validateInput(endY)) {
      resultDisplay.textContent = 'Please enter a number between 0-511 for all fields.';
      return;
    }

    const { preDistance, postDistance } = await solver.computeDistance({ x: startX, y: startY }, { x: endX, y: endY });

    resultDisplay.textContent = `Pre-eruption distance: ${preDistance}, Post-eruption distance: ${postDistance}`;
  });
}

function validateInput(value: number): boolean {
  if (isNaN(value) || value < 0 || value > 512) {
    return false;
  }
  return true;
}

main();