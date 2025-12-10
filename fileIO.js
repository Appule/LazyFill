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