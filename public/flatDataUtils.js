/**
 * RGBAのフラットデータからR成分を取り出し、255で割った値の配列を返す関数
 * @param {Uint8ClampedArray} rgbaData - RGBA形式のフラットデータ。長さは4の倍数である必要があります。
 * @returns {number[]} R成分を256で割った値の配列
 */
export function extractNormalizedR(rgbaData) {
  const result = new Array(rgbaData.length / 4);
  for (let i = 0, j = 0; i < rgbaData.length; i += 4, j++) {
    result[j] = rgbaData[i] / 255;
  }
  return result;
}

/**
 * フラットデータを元にcanvasへ円を描画する関数
 * @param {HTMLCanvasElement} canvas - 描画対象のcanvas要素
 * @param {number[]} flatData - フラットデータ配列。各値は0〜1の範囲を想定
 * @param {number} width - グリッドの横方向の要素数
 * @param {number} height - グリッドの縦方向の要素数
 * @returns {void}
 */
export function drawCirclesOnCanvas(canvas, flatData, width, height) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  // 円の直径と間隔
  const diameter = 6;
  const spacing = 12;

  // canvasサイズを設定
  canvas.width = width * spacing;
  canvas.height = height * spacing;

  let index = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = flatData[index++] ?? 0;
      const color = Math.floor(value * 255);
      
      ctx.fillStyle = `rgb(${color}, 0, 0)`;
      ctx.beginPath();
      ctx.arc(
        x * spacing + spacing / 2, // 中心X
        y * spacing + spacing / 2, // 中心Y
        diameter / 2,              // 半径
        0,
        Math.PI * 2
      );
      ctx.fill();
    }
  }
}