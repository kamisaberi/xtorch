#pragma once

#include <torch/torch.h>
#include <tuple>

namespace xt::linalg
{
    /**
     * @brief Computes the Singular Value Decomposition (SVD) of a 2D tensor.
     *
     * This function provides an interface similar to torch.linalg.svd, but uses a
     * stable third-party C++ library (like Eigen) as its backend.
     *
     * Given a matrix A, the SVD is a factorization A = U * S * Vh, where:
     * - U is an orthonormal matrix (left singular vectors).
     * - S is a 1D vector of singular values.
     * - Vh is an orthonormal matrix (conjugate transpose of right singular vectors).
     *
     * @param A A 2D torch::Tensor of shape [m, n].
     * @param full_matrices If true, compute full-sized U and Vh. If false (default),
     *                      compute the "thin" SVD, which is more memory-efficient.
     * @return A std::tuple containing (U, S, Vh).
     *         - U: torch::Tensor of shape [m, k]
     *         - S: torch::Tensor of shape [k]
     *         - Vh: torch::Tensor of shape [k, n]
     *         where k = min(m, n).
     */
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> svd(
        const torch::Tensor& A,
        bool full_matrices = false);


    torch::Tensor pinverse(const torch::Tensor& input, double rcond = 1e-15);

    std::tuple<torch::Tensor, torch::Tensor> eigh(const torch::Tensor& A);

} // namespace xt::linalg
