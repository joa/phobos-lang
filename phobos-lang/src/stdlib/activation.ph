// Activation functions, after `torch.nn.functional`. Each is an expression
// function over a tile or a scalar, so an activation applied to a product
// fuses into the statement that stores it: `y = silu(acc)` reads `acc` once
// and writes `y` once, with no tile in between.

// `max(x, 0)`.
fn relu(x) = max(x, 0.0)

// `relu`, capped at 6.
fn relu6(x) = clamp(x, 0.0, 6.0)

// `x` for a positive `x`, `slope * x` otherwise. `slope` in `[0, 1]`.
fn leaky_relu(x, slope) = max(x, x * slope)

// `x` limited to `[-1, 1]`.
fn hardtanh(x) = clamp(x, -1.0, 1.0)

// `x` for a positive `x`, `alpha * (e^x - 1)` otherwise. `alpha > 0`.
fn elu(x, alpha) = max(x, 0.0) + min(alpha * (exp(x) - 1.0), 0.0)

// `elu` scaled to be self-normalizing, with the constants of Klambauer et al.
fn selu(x) = 1.0507009873554805 * elu(x, 1.6732632423543772)

// `x` for a positive `x`, `alpha * (e^(x / alpha) - 1)` otherwise. `alpha > 0`.
fn celu(x, alpha) = max(x, 0.0) + min(alpha * (exp(x / alpha) - 1.0), 0.0)

// The logistic function, `1 / (1 + e^-x)`.
fn sigmoid(x) = 1.0 / (1.0 + exp(-x))

// `log(sigmoid(x))`, without the overflow of computing it so.
fn log_sigmoid(x) = min(x, 0.0) - log(1.0 + exp(-abs(x)))

// A piecewise-linear `sigmoid`: 0 below -3, 1 above 3.
fn hardsigmoid(x) = clamp(x / 6.0 + 0.5, 0.0, 1.0)

// `x * sigmoid(x)`, also called swish.
fn silu(x) = x / (1.0 + exp(-x))

// `x * sigmoid(beta * x)`.
fn swish(x, beta) = x / (1.0 + exp(-beta * x))

// `x * hardsigmoid(x)`, MobileNetV3's.
fn hardswish(x) = x * hardsigmoid(x)

// `x * Phi(x)`, `Phi` the standard normal's CDF, through `erf`: PyTorch's
// `gelu` with `approximate="none"`.
fn gelu(x) = 0.5 * x * (1.0 + erf(x * 0.7071067811865476))

// `gelu` by its tanh approximation, GPT-2's: PyTorch's `approximate="tanh"`.
fn gelu_tanh(x) = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))

// `x * sigmoid(1.702 * x)`, CLIP's cheaper approximation of `gelu`.
fn quick_gelu(x) = x / (1.0 + exp(-1.702 * x))

// `log(1 + e^x)`, in the form that neither overflows for a large `x` nor
// loses a small one.
fn softplus(x) = max(x, 0.0) + log(1.0 + exp(-abs(x)))

// `x * tanh(softplus(x))`.
fn mish(x) = x * tanh(softplus(x))

// `x / (1 + |x|)`.
fn softsign(x) = x / (1.0 + abs(x))

// `relu(x)^2`, Primer's.
fn squared_relu(x) = square(max(x, 0.0))

// Gated linear units: the gate's activation times the linear branch `up`,
// as a transformer's MLP joins its two projections. The names follow
// Shazeer's "GLU Variants Improve Transformer".
fn swiglu(gate, up) = silu(gate) * up
fn geglu(gate, up) = gelu(gate) * up
fn geglu_tanh(gate, up) = gelu_tanh(gate) * up
fn reglu(gate, up) = relu(gate) * up
