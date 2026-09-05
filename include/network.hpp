// A multi-layer perceptron with hand-derived backpropagation.
//
// Architecture: [input -> hidden(ReLU) -> ... -> output(softmax)]
// Loss: categorical cross-entropy.
// Optimiser: mini-batch stochastic gradient descent.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace nn {

// One fully-connected layer: out = in * W + b
struct Layer {
    Matrix W;   // [inputs, outputs]
    Matrix b;   // [1, outputs]
    Matrix dW;  // gradient of the loss w.r.t. W, same shape
    Matrix db;

    // Values cached during the forward pass because backprop needs them.
    Matrix input;      // what came in  [batch, inputs]
    Matrix preAct;     // W*x + b, before the activation  [batch, outputs]
    Matrix output;     // after the activation
};

class Network {
public:
    // `sizes` is the full topology, e.g. {784, 128, 10}.
    // Every layer but the last uses ReLU; the last uses softmax.
    Network(const std::vector<std::size_t>& sizes, std::uint64_t seed = 42);

    // Forward pass over a batch. X is [batch, inputSize].
    // Returns class probabilities, [batch, outputSize].
    Matrix forward(const Matrix& X);

    // Mean cross-entropy loss. `labels` holds one class index per row.
    double loss(const Matrix& probs, const std::vector<int>& labels) const;

    // Backward pass. Fills dW/db on every layer. Call after forward().
    void backward(const Matrix& probs, const std::vector<int>& labels);

    // Applies the gradients computed by backward().
    void step(double learningRate);

    // One epoch of mini-batch SGD. Returns the mean loss over the epoch.
    double trainEpoch(const Matrix& X, const std::vector<int>& labels,
                      std::size_t batchSize, double learningRate);

    // Fraction of rows classified correctly.
    double accuracy(const Matrix& X, const std::vector<int>& labels);

    // Writes the trained weights as JSON so another runtime can do inference
    // with them — the browser demo loads exactly this file.
    void save(const std::string& path) const;

    std::vector<Layer>& layers() { return layers_; }
    const std::vector<Layer>& layers() const { return layers_; }

private:
    std::vector<Layer> layers_;
    std::uint64_t seed_;
};

// Row-wise softmax, shifted by each row's maximum for numerical stability:
// exp(1000) overflows to infinity, exp(1000 - 1000) does not, and subtracting
// a constant from every logit leaves the result mathematically unchanged.
Matrix softmax(const Matrix& logits);

}  // namespace nn
