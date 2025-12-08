// graph_cut.wgsl

// =========================
// Bindings (group = 0)
// 0: d_u          (RW storage, size N)
// 1: d_u_bar      (RW storage, size N)   // NEW
// 2: d_p          (RW storage, size M)
// 3: d_cap        (RO storage, size M)
// 4: head         (RO storage, size M)
// 5: tail         (RO storage, size M)
// 6: row_ptr      (RO storage, size N+1) // CSR: start indices per node (by tail)
// 7: col_idx      (RO storage, size M)   // CSR: head node per adjacency slot
// 8: rev_edge     (RO storage, size M)   // edge index of reverse edge
// 9: csr_edge_idx (RO storage, size M)   // NEW: edge index per CSR slot
// 10: params      (uniform)
// =========================

struct Params {
  tau   : f32,   // primal step size
  sigma : f32,   // dual step size
  theta : f32,   // extrapolation
  N     : u32,   // node count
  M     : u32,   // edge count
  source: u32,   // source node index
  sink  : u32,   // sink node index
};

@group(0) @binding(0)  var<storage, read_write> d_u        : array<f32>;
@group(0) @binding(1)  var<storage, read_write> d_u_bar    : array<f32>;
@group(0) @binding(2)  var<storage, read_write> d_p        : array<f32>;
@group(0) @binding(3)  var<storage, read>       d_cap      : array<f32>;
@group(0) @binding(4)  var<storage, read>       head       : array<u32>;
@group(0) @binding(5)  var<storage, read>       tail       : array<u32>;
@group(0) @binding(6)  var<storage, read>       row_ptr    : array<u32>;
@group(0) @binding(7)  var<storage, read>       col_idx    : array<u32>;
@group(0) @binding(8)  var<storage, read>       rev_edge   : array<u32>;
@group(0) @binding(9)  var<storage, read>       csr_edge_idx : array<u32>;
@group(0) @binding(10) var<uniform>             params     : Params;

// -------------------------
// Utility: clamp for f32
fn clamp_f32(x: f32, lo: f32, hi: f32) -> f32 {
  return max(lo, min(x, hi));
}

// -------------------------
// Kernel 1: UpdateDual (per-edge)
// p_e <- clamp( p_e + sigma * (u_bar[i] - u_bar[j]), [-cap_e, cap_e] )
@compute @workgroup_size(256)
fn updateDual(@builtin(global_invocation_id) gid : vec3<u32>) {
  let e = gid.x;
  if (e >= params.M) {
    return;
  }

  let i = tail[e];
  let j = head[e];

  let grad   = d_u_bar[i] - d_u_bar[j];
  let p_old  = d_p[e];
  let cap_e  = d_cap[e];
  let p_new  = p_old + params.sigma * grad;

  d_p[e] = clamp_f32(p_new, -cap_e, cap_e);
}

// -------------------------
// Kernel 2: UpdatePrimal (per-node)
// u_i <- clamp( u_i + tau * div_i, [0,1] )
// u_bar_i <- u_i + theta * (u_i - u_old)
// div_i = sum_out (p_e - p_rev_e) using CSR (row_ptr/col_idx) and csr_edge_idx
@compute @workgroup_size(256)
fn updatePrimal(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= params.N) {
    return;
  }

  // Fixed boundary for source/sink
  if (i == params.source || i == params.sink) {
    // keep u and u_bar as-is (or set explicitly if desired)
    return;
  }

  // Compute divergence using outgoing edges of node i
  let start : u32 = row_ptr[i];
  let end   : u32 = row_ptr[i + 1];

  var div : f32 = 0.0;

  // Iterate adjacency slots [start, end)
  var k : u32 = start;
  loop {
    if (k >= end) { break; }

    // CSR slot k corresponds to actual edge index e
    let e : u32 = csr_edge_idx[k];
    // outgoing edge from i -> col_idx[k] (should equal head[e])
    let p_out : f32 = d_p[e];
    let p_in  : f32 = d_p[rev_edge[e]];

    // divergence contribution across this undirected connection
    div = div + (p_out - p_in);

    k = k + 1u;
  }

  // Primal update with clipping to [0, 1]
  let u_old : f32 = d_u[i];
  let u_new : f32 = clamp_f32(u_old + params.tau * div, 0.0, 1.0);
  d_u[i] = u_new;

  // Extrapolation
  d_u_bar[i] = u_new + params.theta * (u_new - u_old);
}
