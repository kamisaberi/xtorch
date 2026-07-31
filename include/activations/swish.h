#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Swish activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the Swish activation function on an input tensor.
     *
     * Swish is a smooth, self-gated activation function defined as:
     * \f$ \text{Swish}(x) = x \cdot \sigma(\beta \cdot x) = \frac{x}{1 + e^{-\beta \cdot x}} \f$
     * When \p beta is 1.0, Swish is identical to the SiLU (Sigmoid Linear Unit) function.
     *
     * @param x The input tensor.
     * @param beta Scaling hyperparameter for the sigmoid gating mechanism (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor swish(const torch::Tensor& x, double beta = 1.0);

    /**
     * @brief A module wrapper for the Swish activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Swish : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the Swish module.
         */
        Swish() = default;

        /**
         * @brief Performs the forward pass for the Swish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}