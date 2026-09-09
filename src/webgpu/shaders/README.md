# Numerical WGSL templates

Edit `templates/*.wgsl.in` for precision-dependent kernels. `templates.json`
maps each template to its scalar and pair-single output names. CMake runs
`scripts/compile_webgpu_shaders.mjs` with Node (18 or later) before embedding the generated
files from the build directory. Generated WGSL is not checked into git.
The browser receives ordinary WGSL; preprocessing adds no solver-loop work.
Kernels with only one arithmetic variant remain ordinary `.wgsl` files.

```wgsl
fn example(a: Real, b: Real, slot: u32) -> Real {
    return add(a, mul(b, literal(0.7071067811865476), slot), slot);
}
```

`Real` becomes `f32` or the two-word `FF` struct. `add`, `sub`, `mul`, `div`,
`neg`, `square`, and `reciprocal` become parenthesized scalar expressions or calls into
`templates/compensated.wgsl`. `scale(a, b, slot)` multiplies a `Real` by an
explicitly scalar `f32`. A rounding slot can be omitted when the enclosing
function already has a variable named `slot`. Slots must be unique within the
workgroup; the library retains the atomic rounding fences required by both
Chrome and Firefox.

`real(x)` promotes a scalar value with a zero low word. `words(high, low)`
constructs a value from separately stored words and discards the low expression
in scalar builds. `literal(decimal)` splits a decimal constant into two f32
words at build time, while preserving the decimal expression in scalar builds.
Use `hi`, `lo`, and `scalar` to extract the leading word, trailing word, or
rounded sum. `normalize` renormalizes a pair; `round` enforces a scalar rounding
fence in paired arithmetic. Both are identities in scalar arithmetic.

Mixed precision is explicit: `Pair` and the `pair_` forms of these intrinsics
always use compensated arithmetic, even in a scalar shader. For example:

```wgsl
var sum: Pair = pair_real(0.0);
sum = pair_add(sum, pair_real(value), slot);
let rounded_sum: f32 = pair_scalar(sum);
```

The Fourier templates share `accumulator.wgsl.in`: scalar sums retain their
Kahan correction, while paired sums retain both arithmetic words. The scalar
inverse transform's selective geometry correction uses explicit `Pair`
products and sums. `residual_norm.wgsl.in` is instantiated once and always uses
paired sums and squares, including the low-low product, for either solver mode.
The generated precision library is included only where needed.

Use `#if PAIRED` / `#else` / `#endif` (or `#if !PAIRED`) for different buffer
layouts or precision-specific algorithms, and `#include "relative/path"` for
shared template fragments. Prefer shared expressions and evaluation order.
These are build-time directives, not WGSL runtime branches. The preprocessor
balances nested arguments and WGSL type constructors and leaves comments and
ordinary WGSL identifiers alone. Intrinsic names are reserved in templates.

Run `node scripts/test_webgpu_shader_templates.mjs` for the preprocessing
contracts, then build and run the browser numerical verification in Chrome and
headless Firefox. Solver changes also need convergence checks for both
precisions and boundary modes; Node/CTest does not execute WGSL on a GPU.
