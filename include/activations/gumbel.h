#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Gumbel activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Applies the Gumbel activation function to the input tensor.
     *
     * Transforms the input tensor according to the Gumbel distribution properties,
     * controlled by the scale parameter \p beta.
     *
     * @param x The input tensor.
     * @param beta Scale or inverse temperature parameter (defaults to 1.0).
     * @return torch::Tensor The transformed output tensor.
     */
    torch::Tensor gumbel(torch::Tensor x, double beta = 1.0);

    /**
     * @brief A module wrapper for the Gumbel activation function.
     *
     * Inherits from `xt::Module` to support dynamic invocation within neural network layers.
     */
    struct Gumbel : xt::Module {
    public:
        /**
         * @brief Default constructor for the Gumbel module.
         */
        Gumbel() = default;

        /**
         * @brief Executes the forward pass for the Gumbel module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}