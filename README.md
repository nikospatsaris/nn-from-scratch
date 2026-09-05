# Neural Network From Scratch (C++)

A feedforward neural network written in plain C++17 — own matrix class, own
backpropagation, no Eigen, no BLAS, no machine learning library of any kind.

**97.63% test accuracy on MNIST**, trained in 25 seconds on a laptop CPU.

**[→ Try it: draw a digit in your browser](https://claude.ai/code/artifact/bb33e7f5-490b-4dc1-a029-1ad8af97686b)**
— the weights this C++ code trained, running as JavaScript. Nothing is sent
anywhere; the whole network is in the page.

I built this to understand what `model.fit()` actually does. Calling a library
is easy; deriving the gradients, getting the transposes the right way round,
and proving the result is correct is where the learning is.

---

## Results

| Dataset | Network | Epochs | Test accuracy | Time |
|---|---|---|---|---|
| MNIST digits | 784 → 64 → 10 | 20 | **97.63%** | 25s |
| Spirals (3 classes) | 2 → 64 → 3 | 12 | **96.89%** | <1s |

```
 epoch   train loss    train acc     test acc    seconds
     1       0.3741       93.31%       93.41%       0.78
    10       0.0762       98.14%       97.29%       0.77
    15       0.0482       98.76%       97.48%       0.78
    20       0.0350       99.22%       97.63%       0.77

Final test accuracy: 97.63%   (total 25.0s)
```

The spiral dataset is two interleaved spirals per class — deliberately not
linearly separable, so reaching 96.89% means the network genuinely learned a
curved decision boundary rather than fitting a line.

## Proving the gradients are right

This is the part I care most about. A backpropagation bug does not announce
itself: get a transpose or a sign wrong and the network still trains, still
converges, and just quietly ends up worse than it should be.

So the maths is verified against the definition of a derivative. Nudge one
weight by ±h, measure how the loss actually moves, and compare that to what
backprop claimed:

```
dL/dw  ≈  ( L(w + h) − L(w − h) ) / 2h
```

```
$ make check

Numerical gradient check
  network   2 -> 16 -> 8 -> 3 (ReLU, softmax)
  samples   60
  step h    1e-05

  parameter              max rel err  verdict
  ---------------------------------------------------
  layer 0 weight (32)      3.20e-09  PASS
  layer 0 bias   (16)      8.74e-07  PASS
  layer 1 weight (43)      3.89e-08  PASS
  layer 1 bias   (8)       9.25e-07  PASS
  layer 2 weight (24)      2.00e-08  PASS
  layer 2 bias   (3)       1.26e-10  PASS

  worst relative error: 9.248e-07 (tolerance 1e-05)
  All gradients match. Backpropagation is correct.
```

The centred difference is used rather than the one-sided `(L(w+h) − L(w))/h`
because its error shrinks with `h²` instead of `h` — several digits more
accuracy for the same step.

## The maths

Forward, for a batch `X`:

```
z₁ = X·W₁ + b₁      a₁ = ReLU(z₁)
z₂ = a₁·W₂ + b₂     p  = softmax(z₂)
L  = −(1/N) Σ log p[i, yᵢ]
```

Backward. Softmax and cross-entropy are differentiated **together** — taken
separately, softmax alone produces a full Jacobian and the algebra is
miserable. Combined, nearly everything cancels and what survives is just
*predicted minus actual*:

```
∂L/∂z₂ = (p − Y) / N
∂L/∂W₂ = a₁ᵀ · ∂L/∂z₂          ∂L/∂b₂ = Σrows ∂L/∂z₂
∂L/∂a₁ = ∂L/∂z₂ · W₂ᵀ
∂L/∂z₁ = ∂L/∂a₁ ⊙ [z₁ > 0]     ← ReLU gate, from the cached pre-activation
∂L/∂W₁ = Xᵀ · ∂L/∂z₁           ∂L/∂b₁ = Σrows ∂L/∂z₁
```

A bias is added to every row of the batch, so its gradient is the sum down the
rows — that is what `sumRows()` is for.

## Details worth knowing

**Cache-friendly multiplication.** The multiply loop is ordered `i-k-j`, not the
textbook `i-j-k`. In `i-j-k` the inner loop walks down a *column* of the right
operand, jumping a full row width in memory each step and missing cache almost
every time. In `i-k-j` both the right operand and the output are walked left to
right, so every cache line loaded is fully used.

Measured on 512×512 doubles with `g++ -O2`: **0.150s → 0.022s, a 6.75×
speedup**, from nothing but the access pattern. Identical arithmetic.

**He initialisation.** Weights start at `N(0, √(2/fan_in))`. ReLU zeroes about
half its inputs, halving the signal variance at every layer; without the factor
of 2, activations shrink toward zero as the network deepens and learning stalls.

**Numerically stable softmax.** Each row has its maximum subtracted before
exponentiating. `exp(1000)` overflows to infinity; `exp(1000 − 1000)` does not,
and subtracting a constant from every logit leaves softmax mathematically
unchanged.

**Static linking.** The Makefile passes `-static-libstdc++ -static-libgcc`.
Without it a MinGW binary can load a different `libstdc++-6.dll` from `PATH` at
run time — Git for Windows ships one — and segfault before `main()` does
anything. Cost me an afternoon; the binary is self-contained now.

## Building and running

Needs only a C++17 compiler and `make`.

```bash
make            # builds bin/train and bin/gradcheck
make check      # verify the gradients
make run        # train (spirals if MNIST isn't downloaded)
```

For MNIST:

```bash
./scripts/get_mnist.sh
./bin/train --epochs 15 --hidden 128 --lr 0.1 --batch 64
```

Options: `--epochs`, `--batch`, `--lr`, `--hidden`, `--limit`, `--spirals`.

## Layout

```
include/matrix.hpp   src/matrix.cpp     matrix type and operations
include/network.hpp  src/network.cpp    layers, forward, backprop, SGD
include/data.hpp     src/data.cpp       MNIST IDX reader, spiral generator
                     src/main.cpp       training CLI
                     src/gradcheck.cpp  numerical gradient verification
web/template.html    web/build.py       browser demo (weights inlined)
```

## The browser demo

`web/` turns the trained weights into a page you can draw on. `--export` writes
them as JSON, `web/build.py` inlines that into a single self-contained HTML
file, and the forward pass is reimplemented in ~25 lines of JavaScript.

Two things make it work:

**Identical predictions.** Running all 10,000 MNIST test digits through the
JavaScript gives 9,763 correct — exactly the number the C++ reports. Same
weights, same arithmetic, same answers.

**MNIST preprocessing.** A raw drawing classifies badly, because MNIST digits
were not raw drawings: each was cropped to its ink, scaled so the longer side
is 20px, and placed in a 28x28 field with its **centre of mass** at the middle.
The page does the same before inference — the small preview shows exactly what
the network receives. Skipping this step costs more accuracy than any other
detail in the project.

```bash
./bin/train --epochs 20 --hidden 64 --export web/weights.json
python web/build.py          # -> web/index.html, ~550 KB, opens anywhere
```

## What I'd add next

- **Momentum or Adam.** Plain SGD is the honest starting point but converges
  slowly; momentum would be a small change with a visible effect.
- **Convolutional layers.** A dense net treats a digit as 784 unrelated numbers
  and throws away the fact that neighbouring pixels are related. Convolution is
  what closes the gap to 99%+ on MNIST.
- **Multi-threaded matmul.** The multiplication is embarrassingly parallel over
  output rows; `std::thread` over row blocks would scale nearly linearly.
- **Regularisation.** Train accuracy reaches 99.22% while test sits at 97.63% —
  that gap is overfitting, and dropout or weight decay would narrow it.
