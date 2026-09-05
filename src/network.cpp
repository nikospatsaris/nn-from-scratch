#include "network.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iomanip>
#include <random>
#include <stdexcept>

namespace nn {

Matrix softmax(const Matrix& logits) {
    Matrix out(logits.rows(), logits.cols());
    for (std::size_t i = 0; i < logits.rows(); ++i) {
        double rowMax = logits(i, 0);
        for (std::size_t j = 1; j < logits.cols(); ++j) {
            rowMax = std::max(rowMax, logits(i, j));
        }
        double sum = 0.0;
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            const double e = std::exp(logits(i, j) - rowMax);
            out(i, j) = e;
            sum += e;
        }
        for (std::size_t j = 0; j < logits.cols(); ++j) {
            out(i, j) /= sum;
        }
    }
    return out;
}

Network::Network(const std::vector<std::size_t>& sizes, std::uint64_t seed)
    : seed_(seed) {
    if (sizes.size() < 2) {
        throw std::invalid_argument("Network needs at least an input and an output size");
    }

    std::mt19937_64 rng(seed);

    for (std::size_t i = 0; i + 1 < sizes.size(); ++i) {
        Layer layer;
        layer.W = Matrix(sizes[i], sizes[i + 1]);
        layer.b = Matrix(1, sizes[i + 1], 0.0);
        layer.dW = Matrix(sizes[i], sizes[i + 1], 0.0);
        layer.db = Matrix(1, sizes[i + 1], 0.0);

        // He initialisation: N(0, sqrt(2 / fan_in)).
        //
        // ReLU zeroes roughly half its inputs, which halves the variance of
        // the signal at every layer. Without the factor of 2 the activations
        // shrink toward zero as the network deepens and learning stalls.
        const double stddev = std::sqrt(2.0 / static_cast<double>(sizes[i]));
        std::normal_distribution<double> dist(0.0, stddev);
        for (std::size_t k = 0; k < layer.W.size(); ++k) {
            layer.W.flat(k) = dist(rng);
        }

        layers_.push_back(std::move(layer));
    }
}

Matrix Network::forward(const Matrix& X) {
    Matrix current = X;

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        Layer& layer = layers_[i];
        layer.input = current;
        layer.preAct = current.matmul(layer.W).addRowVector(layer.b);

        const bool isLast = (i + 1 == layers_.size());
        if (isLast) {
            layer.output = softmax(layer.preAct);
        } else {
            layer.output = layer.preAct.apply([](double v) { return v > 0.0 ? v : 0.0; });
        }
        current = layer.output;
    }
    return current;
}

double Network::loss(const Matrix& probs, const std::vector<int>& labels) const {
    if (probs.rows() != labels.size()) {
        throw std::invalid_argument("loss: batch size does not match label count");
    }
    // Clamp before the log: a probability that underflows to exactly 0 would
    // give -infinity and poison the average.
    const double eps = 1e-12;
    double total = 0.0;
    for (std::size_t i = 0; i < probs.rows(); ++i) {
        const double p = std::max(probs(i, static_cast<std::size_t>(labels[i])), eps);
        total += -std::log(p);
    }
    return total / static_cast<double>(probs.rows());
}

void Network::backward(const Matrix& probs, const std::vector<int>& labels) {
    const std::size_t batch = probs.rows();

    // Softmax and cross-entropy are differentiated together. Taken separately
    // each derivative is messy (softmax alone gives a full Jacobian); combined,
    // almost everything cancels and what is left is simply (predicted - actual).
    Matrix delta = probs;  // [batch, classes]
    for (std::size_t i = 0; i < batch; ++i) {
        delta(i, static_cast<std::size_t>(labels[i])) -= 1.0;
    }
    delta = delta * (1.0 / static_cast<double>(batch));  // mean, not sum

    // Walk backwards through the layers, carrying `delta` = dL/d(pre-activation).
    for (std::size_t idx = layers_.size(); idx-- > 0;) {
        Layer& layer = layers_[idx];

        // Gradients for this layer's parameters.
        layer.dW = layer.input.transpose().matmul(delta);
        layer.db = delta.sumRows();

        if (idx == 0) break;  // no need to propagate past the input

        // Push the error back through the weights, then through the ReLU.
        Matrix dInput = delta.matmul(layer.W.transpose());

        Layer& prev = layers_[idx - 1];
        // d/dx ReLU(x) is 1 where x > 0 and 0 elsewhere. The gate is taken from
        // the pre-activation, which is why forward() cached it.
        Matrix reluMask = prev.preAct.apply([](double v) { return v > 0.0 ? 1.0 : 0.0; });
        delta = dInput.hadamard(reluMask);
    }
}

void Network::step(double learningRate) {
    for (Layer& layer : layers_) {
        for (std::size_t k = 0; k < layer.W.size(); ++k) {
            layer.W.flat(k) -= learningRate * layer.dW.flat(k);
        }
        for (std::size_t k = 0; k < layer.b.size(); ++k) {
            layer.b.flat(k) -= learningRate * layer.db.flat(k);
        }
    }
}

double Network::trainEpoch(const Matrix& X, const std::vector<int>& labels,
                           std::size_t batchSize, double learningRate) {
    const std::size_t n = X.rows();

    // Shuffle the order each epoch so the network never sees the same batch
    // composition twice — otherwise SGD's gradient noise becomes periodic.
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 rng(seed_ + 1);
    std::shuffle(order.begin(), order.end(), rng);
    ++seed_;

    double totalLoss = 0.0;
    std::size_t batches = 0;

    for (std::size_t start = 0; start < n; start += batchSize) {
        const std::size_t end = std::min(start + batchSize, n);
        const std::size_t rows = end - start;

        Matrix batchX(rows, X.cols());
        std::vector<int> batchY(rows);
        for (std::size_t i = 0; i < rows; ++i) {
            const std::size_t src = order[start + i];
            for (std::size_t j = 0; j < X.cols(); ++j) {
                batchX(i, j) = X(src, j);
            }
            batchY[i] = labels[src];
        }

        Matrix probs = forward(batchX);
        totalLoss += loss(probs, batchY);
        backward(probs, batchY);
        step(learningRate);
        ++batches;
    }

    return batches ? totalLoss / static_cast<double>(batches) : 0.0;
}

double Network::accuracy(const Matrix& X, const std::vector<int>& labels) {
    Matrix probs = forward(X);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < probs.rows(); ++i) {
        std::size_t best = 0;
        for (std::size_t j = 1; j < probs.cols(); ++j) {
            if (probs(i, j) > probs(i, best)) best = j;
        }
        if (static_cast<int>(best) == labels[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(probs.rows());
}

void Network::save(const std::string& path) const {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write " + path);

    // Six significant digits is far more than the network needs — the rounding
    // error is orders of magnitude below anything that could change a
    // prediction — and it keeps the file small enough to embed in a web page.
    out << std::setprecision(6);
    out << "{\n  \"layers\": [\n";

    for (std::size_t i = 0; i < layers_.size(); ++i) {
        const Layer& layer = layers_[i];
        const bool isLast = (i + 1 == layers_.size());

        out << "    {\n";
        out << "      \"in\": " << layer.W.rows() << ",\n";
        out << "      \"out\": " << layer.W.cols() << ",\n";
        out << "      \"activation\": \"" << (isLast ? "softmax" : "relu") << "\",\n";

        out << "      \"W\": [";
        for (std::size_t k = 0; k < layer.W.size(); ++k) {
            if (k) out << ',';
            out << layer.W.flat(k);
        }
        out << "],\n";

        out << "      \"b\": [";
        for (std::size_t k = 0; k < layer.b.size(); ++k) {
            if (k) out << ',';
            out << layer.b.flat(k);
        }
        out << "]\n    }";
        if (!isLast) out << ',';
        out << '\n';
    }

    out << "  ]\n}\n";
    if (!out) throw std::runtime_error("failed while writing " + path);
}

}  // namespace nn
