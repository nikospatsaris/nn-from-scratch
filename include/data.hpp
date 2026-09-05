// Dataset loading.
//
// MNIST when the IDX files are present, and a generated spiral dataset
// otherwise, so the project trains and demonstrates itself with no download.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace nn {

struct Dataset {
    Matrix X;                 // [samples, features]
    std::vector<int> y;       // class index per sample
    std::size_t numClasses = 0;
};

// Reads MNIST IDX files (already gunzipped). Throws on a missing or malformed
// file. Pixels are scaled to [0, 1]; `limit` of 0 means read everything.
Dataset loadMnist(const std::string& imagePath, const std::string& labelPath,
                  std::size_t limit = 0);

// Two interleaved spirals per class: not linearly separable, so a network that
// gets this right must have learned a curved boundary rather than a line.
Dataset makeSpirals(std::size_t pointsPerClass, std::size_t numClasses,
                    std::uint64_t seed = 7);

bool fileExists(const std::string& path);

}  // namespace nn
