#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SwiGLU (Swish Gated Linear Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SwiGLU (Swish Gated Linear Unit) activation on an input tensor.
     *
     * SwiGLU splits the input tensor into two equal halves along dimension \p dim,
     * applies the Swish activation function (parameterized by \p beta) to the first half,
     * and multiplies it element-wise by the second half:
     * \f$ \text{SwiGLU}(x_1, x_2) = (x_1 \cdot \sigma(\beta \cdot x_1)) \otimes x_2 \f$
     *
     * @param x The input tensor.
     * @param dim The dimension along which to split the input tensor into two halves (defaults to 1).
     * @param beta Scaling hyperparameter for the Swish gating function (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor swiglu(const torch::Tensor& x, int64_t dim = 1, double beta = 1.0);

    /**
     * @brief A module wrapper for the SwiGLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SwiGLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the SwiGLU module.
         */
        SwiGLU() = default;

        /**
         * @brief Performs the forward pass for the SwiGLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}