#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Mish activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Mish activation function on an input tensor.
     *
     * Mish is a self-gated, smooth, non-monotonic activation function defined as:
     * \f$ \text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x)) \f$
     *
     * @param x The input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor mish(torch::Tensor x);

    /**
     * @brief A module wrapper for the Mish activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct Mish : xt::Module {
    public:
        /**
         * @brief Default constructor for the Mish module.
         */
        Mish() = default;

        /**
         * @brief Performs the forward pass for the Mish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}