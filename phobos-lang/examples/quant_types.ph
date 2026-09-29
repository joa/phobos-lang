// Narrow element types end to end: i8 weights and bf16 scales, converted and
// combined in f32, the way a Q8_0 matvec does its arithmetic.
kernel quant_types(W: tensor<i8>[M, N],
                   S: tensor<bf16>[M, N],
                   H: tensor<f16>[M, N],
                   C: tensor<f32>[M, N]) {
  let pm = program_id(0)
  let pn = program_id(1)

  let w = W[pm * 32 :+ 32, pn * 32 :+ 32]
  let s = S[pm * 32 :+ 32, pn * 32 :+ 32]
  let h = H[pm * 32 :+ 32, pn * 32 :+ 32]

  // i8 to f32 sign-extends and converts. bf16 to f32 is a shift on any arch.
  var wf: tile<f32>[32, 32] = f32(w)
  var sf: tile<f32>[32, 32] = f32(s)

  // f16 and bf16 have no direct conversion, so this sum meets at f32.
  var mixed: tile<f32>[32, 32] = h + s

  var out: tile<f32>[32, 32] = wf * sf
  out = out + mixed

  // Round to bf16 and back, to exercise the narrowing conversion.
  var narrow: tile<bf16>[32, 32] = bf16(out)
  C[pm * 32 :+ 32, pn * 32 :+ 32] = f32(narrow)
}
