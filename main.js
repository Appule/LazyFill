import { getImageDataFromFileInput, drawFlatArrayToCanvas } from './fileIO.js';
import { imageProc } from './imageProc.js';

export async function testImgProc() {
  const fileInput = document.getElementById('fileInput');
  const { data, width, height } = await getImageDataFromFileInput(fileInput);
  const luma = await imageProc(data, width, height);
  drawFlatArrayToCanvas(document.getElementById('canvas'), luma, width, height);
}