struct Params {
  width: u32,
  height: u32,
  length: u32,
  scl: f32,
}

@group(0) @binding(0)
var<uniform> params: Params;