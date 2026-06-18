#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ShiLU (Shifted Linear Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the ShiLU (Shifted Linear Unit) activation function on an input tensor.
     *
     * ShiLU applies a scaled and shifted linear unit transformation to the input tensor \p x
     * using scale factor \p a and shift offset \p b:
     * \f$ \text{ShiLU}(x) = a \cdot \max(0, x) + b \f$
     *
     * @param x The input tensor.
     * @param a Scaling factor for the positive linear region (defaults to 1.0).
     * @param b Shift offset hyperparameter (defaults to 0.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor shilu(const torch::Tensor& x, double a = 1.0, double b = 0.0);

    /**
     * @brief A module wrapper for the ShiLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct ShiLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the ShiLU module.
         */
        ShiLU() = default;

        /**
         * @brief Performs the forward pass for the ShiLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}