// Numerical gradient checking — the proof that backpropagation is correct.
//
// Backprop computes dL/dw analytically. Calculus also gives a slow but
// assumption-free way to get the same number: nudge one weight by ±h and
// measure how the loss actually moves.
//
//     dL/dw  ~=  ( L(w + h) - L(w - h) ) / 2h
//
// The centred form is used rather than the one-sided (L(w+h) - L(w))/h
// because its error shrinks with h^2 instead of h — several digits more
// accuracy for the same step size.
//
// If the hand-derived gradient and the measured one agree to ~1e-7, the
// derivation is right. If a sign or a transpose were wrong, the network would
// still train to a mediocre accuracy and look fine; this is the test that
// actually catches it.

#include <cmath>
#include <cstdio>
#include <vector>

#include "data.hpp"
#include "matrix.hpp"
#include "network.hpp"

namespace {

double relativeError(double analytic, double numeric) {
    const double denom = std::max(1e-12, std::fabs(analytic) + std::fabs(numeric));
    return std::fabs(analytic - numeric) / denom;
}

struct Result {
    double maxRelError = 0.0;
    std::size_t checked = 0;
};

Result checkLayer(nn::Network& net, std::size_t layerIndex, bool checkBias,
                  const nn::Matrix& X, const std::vector<int>& y,
                  std::size_t maxParams, double h) {
    Result result;

    // Analytic gradients for the current parameters.
    nn::Matrix probs = net.forward(X);
    net.backward(probs, y);

    nn::Matrix& param = checkBias ? net.layers()[layerIndex].b
                                  : net.layers()[layerIndex].W;
    const nn::Matrix grad = checkBias ? net.layers()[layerIndex].db
                                      : net.layers()[layerIndex].dW;

    const std::size_t total = param.size();
    const std::size_t stride = total > maxParams ? total / maxParams : 1;

    for (std::size_t i = 0; i < total; i += stride) {
        const double original = param.flat(i);

        param.flat(i) = original + h;
        const double lossPlus = net.loss(net.forward(X), y);

        param.flat(i) = original - h;
        const double lossMinus = net.loss(net.forward(X), y);

        param.flat(i) = original;  // restore before moving on

        const double numeric = (lossPlus - lossMinus) / (2.0 * h);
        const double analytic = grad.flat(i);

        result.maxRelError = std::max(result.maxRelError, relativeError(analytic, numeric));
        ++result.checked;
    }
    return result;
}

}  // namespace

int main() {
    // A small network and a small batch: gradient checking runs two full
    // forward passes per parameter, so it is far too slow for a real model.
    nn::Dataset ds = nn::makeSpirals(20, 3);
    nn::Network net({2, 16, 8, 3}, /*seed=*/1234);

    // h is a compromise. Too large and the finite difference stops
    // approximating the derivative; too small and floating-point cancellation
    // in (L+ - L-) destroys the answer. 1e-5 sits near the sweet spot for
    // doubles.
    const double h = 1e-5;
    // 1e-5 is the conventional pass mark for centred differences in double
    // precision. In practice this network lands around 1e-7 to 1e-9; the
    // margin is there so the check does not report a false failure on a
    // different compiler or floating-point setting.
    const double tolerance = 1e-5;

    std::printf("Numerical gradient check\n");
    std::printf("  network   2 -> 16 -> 8 -> 3 (ReLU, softmax)\n");
    std::printf("  samples   %zu\n", ds.y.size());
    std::printf("  step h    %g\n\n", h);
    std::printf("  %-22s %10s  %s\n", "parameter", "max rel err", "verdict");
    std::printf("  ---------------------------------------------------\n");

    double worst = 0.0;
    bool allPassed = true;

    for (std::size_t layer = 0; layer < net.layers().size(); ++layer) {
        for (int biasPass = 0; biasPass < 2; ++biasPass) {
            const bool isBias = (biasPass == 1);
            Result r = checkLayer(net, layer, isBias, ds.X, ds.y, /*maxParams=*/40, h);

            char name[64];
            std::snprintf(name, sizeof(name), "layer %zu %s (%zu)", layer,
                          isBias ? "bias  " : "weight", r.checked);

            const bool passed = r.maxRelError < tolerance;
            allPassed = allPassed && passed;
            worst = std::max(worst, r.maxRelError);

            std::printf("  %-22s %10.2e  %s\n", name, r.maxRelError,
                        passed ? "PASS" : "FAIL");
        }
    }

    std::printf("\n  worst relative error: %.3e (tolerance %.0e)\n", worst, tolerance);
    if (allPassed) {
        std::printf("  All gradients match. Backpropagation is correct.\n");
        return 0;
    }
    std::printf("  MISMATCH — the derivation is wrong somewhere.\n");
    return 1;
}
