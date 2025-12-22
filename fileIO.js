/**
 * ファイル入力要素から画像ファイルを読み込み、ピクセルデータ(RGBA)のフラット配列を返します。
 * @param {HTMLInputElement} fileInput - `type="file"` の input 要素。
 * @returns {Promise<{data: Uint8ClampedArray, width: number, height: number}>} 画像の幅、高さ、およびRGBAピクセルデータを格納したUint8ClampedArrayを含むオブジェクトを解決するPromise。ファイルが選択されていない場合や、画像の読み込みに失敗した場合はPromiseがrejectされます。
 */
export function getImageDataFromFileInput(fileInput) {
  return new Promise((resolve, reject) => {
    const file = fileInput.files?.[0];
    if (!file) {
      reject(new Error("ファイルが選択されていません。"));
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        resolve({
          data: imageData.data,
          width: imageData.width,
          height: imageData.height,
        });
      };
      img.onerror = () => reject(new Error("画像の読み込みに失敗しました。"));
      img.src = e.target.result;
    };
    reader.onerror = () => reject(new Error("ファイルの読み込みに失敗しました。"));
    reader.readAsDataURL(file);
  });
}

/**
 * RGBAピクセルデータのフラット配列をCanvasに描画します。
 * @param {HTMLCanvasElement} canvas - 描画対象の canvas 要素。
 * @param {Uint8ClampedArray} flatRgbaArray - 描画するRGBAピクセルデータのフラット配列。
 * @param {number} width - 画像の幅。
 * @param {number} height - 画像の高さ。
 */
export function drawFlatArrayToCanvas(canvas, flatRgbaArray, width, height) {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  imageData.data.set(flatRgbaArray);
  ctx.putImageData(imageData, 0, 0);
}

/**
 * Float32 の輝度値配列を Canvas にグレースケール画像として描画します。
 * @param {HTMLCanvasElement} canvas - 描画対象の canvas 要素。
 * @param {Float32Array} float32Array - 輝度値（0〜1 または 0〜255）のフラット配列。
 * @param {number} width - 画像の幅。
 * @param {number} height - 画像の高さ。
 * @param {boolean} normalized - true の場合は 0〜1 を 0〜255 に正規化して扱う
 */
export function drawFloatArrayToCanvas(canvas, float32Array, width, height, normalized = true) {
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  const data = imageData.data; // Uint8ClampedArray

  for (let i = 0; i < float32Array.length; i++) {
    // 輝度値を 0〜255 に変換
    let v = float32Array[i];
    if (normalized) {
      v = Math.min(Math.max(v * 255, 0), 255);
    } else {
      v = Math.min(Math.max(v, 0), 255);
    }

    const idx = i * 4;
    data[idx] = v;     // R
    data[idx + 1] = v; // G
    data[idx + 2] = v; // B
    data[idx + 3] = 255; // A（不透明）
  }

  ctx.putImageData(imageData, 0, 0);
}
