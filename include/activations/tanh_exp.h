#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the TanhExp activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the TanhExp activation function on an input tensor.
     *
     * TanhExp is a smooth, self-gated activation function defined as:
     * \f$ \text{TanhExp}(x) = x \cdot \tanh(e^x) \f$
     *
     * @param x The input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor tanh_exp(const torch::Tensor& x);

    /**
     * @brief A module wrapper for the TanhExp activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct TanhExp : xt::Module {
    public:
        /**
         * @brief Default constructor for the TanhExp module.
         */
        TanhExp() = default;

        /**
         * @brief Performs the forward pass for the TanhExp module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}