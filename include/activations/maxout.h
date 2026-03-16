#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Maxout activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Maxout activation function on an input tensor.
     *
     * Maxout splits the input tensor along dimension \p dim into \p num_pieces
     * groups and calculates the element-wise maximum across each group:
     * \f$ \text{Maxout}(x)_{i} = \max_{j \in [1, k]} x_{i, j} \f$
     *
     * @param x The input tensor.
     * @param num_pieces The number of pieces/groups to take the maximum across.
     * @param dim The dimension along which to split and perform max reduction (defaults to 1).
     * @return torch::Tensor The reduced output tensor.
     */
    torch::Tensor maxout(const torch::Tensor& x, int64_t num_pieces, int64_t dim = 1) ;

    /**
     * @brief A module wrapper for the Maxout activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Maxout : xt::Module {
    public:
        /**
         * @brief Default constructor for the Maxout module.
         */
        Maxout() = default;

        /**
         * @brief Performs the forward pass for the Maxout module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}