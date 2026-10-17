// Matrix products over tensors, one output tile at a time. Each is the
// k-loop a GEMM is made of, so the compiler stages, pipelines and lowers
// it to tensor cores exactly as the loop written out: the kernel around it
// picks the tile, the grid and the attributes, and the epilogue after it
// fuses whatever the caller writes next.

// `acc += A[m0 :+ BM, :] B[:, n0 :+ BN]`, `BK` columns of `A` a step.
// `A` is `[M, K]`, `B` is `[K, N]` and `acc` is `[BM, BN]`.
fn matmul(acc, A, B, m0, n0, BK) {
  for kt in range(0, extent(A, 1), BK) {
    var a = A[m0 :+ extent(acc, 0), kt :+ BK]
    var b = B[kt :+ BK, n0 :+ extent(acc, 1)]
    acc += dot(a, b)
  }
}

// `acc += A[m0 :+ BM, :] B[n0 :+ BN, :]^T`, against a `B` stored `[N, K]`,
// as a linear layer's weight is: `matmul_t(acc, X, W, m0, n0, BK)` is
// `torch.nn.functional.linear` without the bias.
fn matmul_t(acc, A, B, m0, n0, BK) {
  for kt in range(0, extent(A, 1), BK) {
    var a = A[m0 :+ extent(acc, 0), kt :+ BK]
    var b = B[n0 :+ extent(acc, 1), kt :+ BK]
    acc += dot_t(a, b)
  }
}
