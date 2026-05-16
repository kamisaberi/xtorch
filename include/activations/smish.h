#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Smish activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Smish activation function on an input tensor.
     *
     * Smish is a smooth, non-monotonic activation function parameterized by scaling factors \p alpha and \p beta:
     * \f$ \text{Smish}(x) = \alpha \cdot x \cdot \tanh(\text{softplus}(\beta \cdot x)) \f$
     *
     * @param x The input tensor.
     * @param alpha Output scaling parameter (defaults to 1.0).
     * @param beta Input scaling parameter for softplus (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor smish(const torch::Tensor& x, double alpha = 1.0, double beta = 1.0);

    /**
     * @brief A module wrapper for the Smish activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Smish : xt::Module {
    public:
        /**
         * @brief Default constructor for the Smish module.
         */
        Smish() = default;

        /**
         * @brief Performs the forward pass for the Smish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}