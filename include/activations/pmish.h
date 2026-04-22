#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the PMish (Parametric Mish) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the PMish (Parametric Mish) activation function on an input tensor.
     *
     * PMish is a parameterized variant of the Mish activation function, defined as:
     * \f$ \text{PMish}(x) = x \cdot \tanh(\alpha \cdot \text{softplus}(\beta \cdot x)) \f$
     *
     * @param x The input tensor.
     * @param alpha Scaling parameter applied to softplus before tanh (defaults to 1.0).
     * @param beta Scaling parameter applied to input \p x inside softplus (defaults to 0.5).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor pmish(const torch::Tensor& x, double alpha = 1.0, double beta = 0.5);

    /**
     * @brief A module wrapper for the PMish activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct PMish : xt::Module {
    public:
        /**
         * @brief Default constructor for the PMish module.
         */
        PMish() = default;

        /**
         * @brief Performs the forward pass for the PMish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}