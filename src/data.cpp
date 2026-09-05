#include "data.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>

namespace nn {

bool fileExists(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// IDX stores its header integers big-endian, so on x86 the bytes have to be
// reversed after reading.
static std::uint32_t readBigEndianU32(std::istream& in) {
    unsigned char bytes[4];
    in.read(reinterpret_cast<char*>(bytes), 4);
    if (!in) throw std::runtime_error("IDX: unexpected end of file in header");
    return (static_cast<std::uint32_t>(bytes[0]) << 24) |
           (static_cast<std::uint32_t>(bytes[1]) << 16) |
           (static_cast<std::uint32_t>(bytes[2]) << 8) |
           static_cast<std::uint32_t>(bytes[3]);
}

Dataset loadMnist(const std::string& imagePath, const std::string& labelPath,
                  std::size_t limit) {
    std::ifstream images(imagePath, std::ios::binary);
    std::ifstream labels(labelPath, std::ios::binary);
    if (!images) throw std::runtime_error("cannot open image file: " + imagePath);
    if (!labels) throw std::runtime_error("cannot open label file: " + labelPath);

    if (readBigEndianU32(images) != 2051) {
        throw std::runtime_error("bad magic number in image file (expected 2051)");
    }
    const std::uint32_t imageCount = readBigEndianU32(images);
    const std::uint32_t rows = readBigEndianU32(images);
    const std::uint32_t cols = readBigEndianU32(images);

    if (readBigEndianU32(labels) != 2049) {
        throw std::runtime_error("bad magic number in label file (expected 2049)");
    }
    const std::uint32_t labelCount = readBigEndianU32(labels);

    if (imageCount != labelCount) {
        throw std::runtime_error("image/label count mismatch");
    }

    std::size_t n = imageCount;
    if (limit > 0 && limit < n) n = limit;

    const std::size_t features = static_cast<std::size_t>(rows) * cols;
    Dataset ds;
    ds.X = Matrix(n, features);
    ds.y.resize(n);
    ds.numClasses = 10;

    std::vector<unsigned char> buffer(features);
    for (std::size_t i = 0; i < n; ++i) {
        images.read(reinterpret_cast<char*>(buffer.data()),
                    static_cast<std::streamsize>(features));
        if (!images) throw std::runtime_error("IDX: truncated image data");
        for (std::size_t j = 0; j < features; ++j) {
            // Scale to [0, 1]. Raw 0-255 inputs would make the first layer's
            // pre-activations huge and stall training from the first step.
            ds.X(i, j) = static_cast<double>(buffer[j]) / 255.0;
        }

        unsigned char label = 0;
        labels.read(reinterpret_cast<char*>(&label), 1);
        if (!labels) throw std::runtime_error("IDX: truncated label data");
        ds.y[i] = static_cast<int>(label);
    }

    return ds;
}

Dataset makeSpirals(std::size_t pointsPerClass, std::size_t numClasses,
                    std::uint64_t seed) {
    Dataset ds;
    const std::size_t n = pointsPerClass * numClasses;
    ds.X = Matrix(n, 2);
    ds.y.resize(n);
    ds.numClasses = numClasses;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> noise(0.0, 0.02);

    std::size_t idx = 0;
    for (std::size_t c = 0; c < numClasses; ++c) {
        for (std::size_t i = 0; i < pointsPerClass; ++i) {
            const double r = static_cast<double>(i) / static_cast<double>(pointsPerClass);
            const double t = static_cast<double>(c) * 4.0 +
                             r * 4.0 +
                             noise(rng) * 10.0;
            ds.X(idx, 0) = r * std::sin(t);
            ds.X(idx, 1) = r * std::cos(t);
            ds.y[idx] = static_cast<int>(c);
            ++idx;
        }
    }
    return ds;
}

}  // namespace nn
