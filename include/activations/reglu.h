#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ReGLU (Rectified Linear Gated Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the ReGLU (Rectified Linear Gated Unit) activation on an input tensor.
     *
     * ReGLU splits the input tensor into two equal halves along dimension \p dim,
     * applies the ReLU activation function to the first half, and multiplies it element-wise
     * by the second half:
     * \f$ \text{ReGLU}(x_1, x_2) = \text{ReLU}(x_1) \otimes x_2 \f$
     *
     * @param x The input tensor.
     * @param dim The dimension along which to split the input tensor (defaults to 1).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor reglu(const torch::Tensor& x, int64_t dim = 1);

    /**
     * @brief A module wrapper for the ReGLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct ReGLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the ReGLU module.
         */
        ReGLU() = default;

        /**
         * @brief Performs the forward pass for the ReGLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}