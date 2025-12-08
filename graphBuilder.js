/**
 * エッジを追加する関数
 * @param {number} tail - 始点ノード
 * @param {number} head - 終点ノード
 * @param {Array} edges - エッジリスト (参照渡し)
 */
export function addEdge(tail, head, edges) {
  edges.push({ tail, head });
}

/**
 * 格子状グラフを構築する関数
 * @param {number} rows - 行数
 * @param {number} cols - 列数
 * @returns {{N:number, edges:Array}} - ノード数とエッジリスト
 */
export function buildGridGraph(rows, cols) {
  const edges = [];
  const N = rows * cols;

  // ノード番号は (r * cols + c)
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const node = r * cols + c;

      // 右方向の隣接ノード
      if (c + 1 < cols) {
        const right = r * cols + (c + 1);
        addEdge(node, right, edges);
        addEdge(right, node, edges); // 逆方向も追加
      }

      // 下方向の隣接ノード
      if (r + 1 < rows) {
        const down = (r + 1) * cols + c;
        addEdge(node, down, edges);
        addEdge(down, node, edges); // 逆方向も追加
      }
    }
  }

  return { N, edges };
}

import { createBuffer } from './wgpuUtils.js';

/**
 * グラフカット用のバッファ群をまとめて生成する関数
 */
export function createGraphBuffers(device, N, M, initData = {}) {
  // usage を調整: initData があるものは COPY_SRC も付ける
  const makeBuffer = (arr, hasInit) => {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | (hasInit ? GPUBufferUsage.COPY_SRC : 0);
    return createBuffer(device, arr, usage);
  };

  const d_u = new Float32Array(initData.d_u || N);
  const d_u_bar = new Float32Array(initData.d_u_bar || N);
  const d_p = new Float32Array(initData.d_p || M);
  const d_cap = new Float32Array(initData.d_cap || M);
  const head = new Uint32Array(initData.head || M);
  const tail = new Uint32Array(initData.tail || M);
  const row_ptr = new Uint32Array(initData.row_ptr || (N + 1));
  const col_idx = new Uint32Array(initData.col_idx || M);
  const rev_edge = new Uint32Array(initData.rev_edge || M);
  const csr_edge_idx = new Uint32Array(initData.csr_edge_idx || M);

  return {
    d_u: makeBuffer(d_u, !!initData.d_u),
    d_u_bar: makeBuffer(d_u_bar, !!initData.d_u_bar),
    d_p: makeBuffer(d_p, !!initData.d_p),
    d_cap: makeBuffer(d_cap, !!initData.d_cap),
    head: makeBuffer(head, !!initData.head),
    tail: makeBuffer(tail, !!initData.tail),
    row_ptr: makeBuffer(row_ptr, !!initData.row_ptr),
    col_idx: makeBuffer(col_idx, !!initData.col_idx),
    rev_edge: makeBuffer(rev_edge, !!initData.rev_edge),
    csr_edge_idx: makeBuffer(csr_edge_idx, !!initData.csr_edge_idx), // 追加
  };
}

/**
 * CSR形式のグラフ構造を構築する関数
 * @param {number} N - ノード数
 * @param {Array<{tail:number, head:number}>} edges - エッジリスト
 * @returns {{row_ptr: Uint32Array, col_idx: Uint32Array, rev_edge: Uint32Array}}
 */
export function buildCSR(N, edges) {
  const M = edges.length;
  const row_ptr = new Uint32Array(N + 1);
  const col_idx = new Uint32Array(M);
  const rev_edge = new Uint32Array(M);
  const csr_edge_idx = new Uint32Array(M);

  // 出次数カウント
  for (let e = 0; e < M; e++) {
    row_ptr[edges[e].tail + 1]++;
  }

  // 累積和
  for (let i = 1; i <= N; i++) row_ptr[i] += row_ptr[i - 1];

  // 一時カウンタ
  const counter = new Uint32Array(N);

  for (let e = 0; e < M; e++) {
    const { tail, head } = edges[e];
    const pos = row_ptr[tail] + counter[tail];
    col_idx[pos] = head;
    csr_edge_idx[pos] = e;
    counter[tail]++;
  }

  // rev_edge を別ループで構築（O(M^2) のままでも良いが改善可能）
  for (let i = 0; i < M; i++) {
    const { tail, head } = edges[i];
    rev_edge[i] = 0;
    for (let j = 0; j < M; j++) {
      if (edges[j].tail === head && edges[j].head === tail) {
        rev_edge[i] = j;
        break;
      }
    }
  }

  return { row_ptr, col_idx, rev_edge, csr_edge_idx };
}


/**
 * Params uniform buffer を生成する関数
 * @param {GPUDevice} device - WebGPU デバイス
 * @param {Object} values - { tau, sigma, theta, N, M, source, sink }
 * @returns {GPUBuffer} - uniform buffer
 */
export function createParamsBuffer(device, values) {
  // WGSL struct Params に対応する順序で並べる
  const data = new Float32Array([
    values.tau ?? 0.1,
    values.sigma ?? 0.1,
    values.theta ?? 0.5,
    values.N ?? 0,
    values.M ?? 0,
    values.source ?? 0,
    values.sink ?? 0,
  ]);

  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    mappedAtCreation: true,
  });

  new Float32Array(buffer.getMappedRange()).set(data);
  buffer.unmap();

  return buffer;
}