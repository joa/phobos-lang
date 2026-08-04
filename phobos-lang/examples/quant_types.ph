// Narrow element types end to end: an int8 weight tensor and a bf16 scale
// tensor converted into an f32 accumulation.
//
// This is the shape a Q8_0 dequantizing matvec has: the weights arrive as
// signed bytes, one scale covers a block of them, and the arithmetic happens
// in f32 after conversion.
kernel quant_types(W: tensor<i8>[M, N],
                   S: tensor<bf16>[M, N],
                   H: tensor<f16>[M, N],
                   C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)

  let w = W[pm * 32 :+ 32, pn * 32 :+ 32]
  let s = S[pm * 32 :+ 32, pn * 32 :+ 32]
  let h = H[pm * 32 :+ 32, pn * 32 :+ 32]

  // i8 -> f32 sign-extends and converts; bf16 -> f32 is a shift on any arch.
  var wf: tile<f32>[32, 32] = f32(w)
  var sf: tile<f32>[32, 32] = f32(s)

  // f16 and bf16 have no direct conversion between them: this meets at f32.
  var mixed: tile<f32>[32, 32] = h + s

  var out: tile<f32>[32, 32] = wf * sf
  out = out + mixed

  // Round the f32 result back down to bf16 and out again, exercising the
  // narrowing direction.
  var narrow: tile<bf16>[32, 32] = bf16(out)
  C[pm * 32 :+ 32, pn * 32 :+ 32] = f32(narrow)
}
