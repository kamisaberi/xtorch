#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Phish activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Phish activation function on an input tensor.
     *
     * Phish is a smooth, non-monotonic activation function defined as:
     * \f$ \text{Phish}(x) = a \cdot x \cdot \tanh(\text{GELU}(b \cdot x)) \f$
     *
     * @param x The input tensor.
     * @param a Output scaling hyperparameter (defaults to 1.0).
     * @param b Input scaling hyperparameter for GELU (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor phish(const torch::Tensor& x, double a = 1.0, double b = 1.0);

    /**
     * @brief A module wrapper for the Phish activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Phish : xt::Module {
    public:
        /**
         * @brief Default constructor for the Phish module.
         */
        Phish() = default;

        /**
         * @brief Performs the forward pass for the Phish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}