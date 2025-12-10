import { getImageDataFromFileInput, drawFlatArrayToCanvas } from './fileIO.js';
import { rgbaToGrayscale } from './imageProc.js';

export async function testImgProc() {
  const fileInput = document.getElementById('fileInput');
  const { data, width, height } = await getImageDataFromFileInput(fileInput);
  const luma = await rgbaToGrayscale(data);
  drawFlatArrayToCanvas(document.getElementById('canvas'), luma, width, height);
}