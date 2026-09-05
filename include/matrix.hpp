// A minimal dense matrix, written from scratch.
//
// No Eigen, no BLAS, no external maths library. Storage is a flat row-major
// vector because that keeps a whole row contiguous in memory, which is what
// makes the multiplication loop below cache-friendly.

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace nn {

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}
    Matrix(std::size_t rows, std::size_t cols, double fill = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, fill) {}

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    std::size_t size() const { return data_.size(); }

    double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    // Flat access, used by the gradient checker to walk every parameter.
    double& flat(std::size_t i) { return data_[i]; }
    double flat(std::size_t i) const { return data_[i]; }

    Matrix matmul(const Matrix& rhs) const;
    Matrix transpose() const;

    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix& rhs) const;
    Matrix operator*(double scalar) const;

    // Element-wise product. Named rather than overloaded on `*` so it can
    // never be confused with matrix multiplication at a call site.
    Matrix hadamard(const Matrix& rhs) const;

    // Adds a 1 x cols row vector to every row — how a bias is applied to a
    // whole mini-batch at once.
    Matrix addRowVector(const Matrix& rowVec) const;

    // Collapses a batch down to a single row of column sums. This is the
    // gradient of a bias: the bias touched every row, so its gradient is the
    // sum of the gradients of all of them.
    Matrix sumRows() const;

    Matrix apply(const std::function<double(double)>& fn) const;

    void fill(double value);

private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

}  // namespace nn
