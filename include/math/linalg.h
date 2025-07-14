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


    /**
     * @brief Computes the eigenvalues and right eigenvectors of a GENERAL square matrix.
     *
     * This function uses a general-purpose solver. The eigenvalues and eigenvectors
     * can be complex even if the input matrix is real.
     *
     * @param A A 2D square torch::Tensor of shape [n, n].
     * @return A std::tuple containing (eigenvalues, eigenvectors) as complex tensors.
     */
    std::tuple<torch::Tensor, torch::Tensor> eig(const torch::Tensor& A);

    /**
     * @brief Computes the eigenvalues and eigenvectors of a SYMMETRIC/HERMITIAN matrix.
     *
     * This function uses a specialized, more efficient solver for symmetric matrices.
     * The eigenvalues are guaranteed to be real.
     *
     * @param A A 2D symmetric square torch::Tensor of shape [n, n].
     * @param UPLO Specifies whether to use the upper ('U') or lower ('L') triangular
     *             part of the input matrix. Defaults to 'L'.
     * @return A std::tuple containing (eigenvalues, eigenvectors) as REAL tensors.
     */
    std::tuple<torch::Tensor, torch::Tensor> eigh(
        const torch::Tensor& A,
        const std::string& UPLO = "L");
} // namespace xt::linalg
