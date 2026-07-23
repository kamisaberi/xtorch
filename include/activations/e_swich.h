/**
* @file e_swish.h
 * @brief Declaration of the E-Swish activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the E-Swish activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param beta Parameter controlling the non-linearity scale (default: 1.25).
     * @return torch::Tensor Output tensor with the E-Swish activation applied.
     */
    torch::Tensor e_swish(torch::Tensor x, double beta = 1.25);

    /**
     * @struct ESwish
     * @brief High-level module wrapper for the E-Swish activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct ESwish : xt::Module {
    public:
        /**
         * @brief Default constructor for ESwish.
         */
        ESwish() = default;

        /**
         * @brief Forward pass for the ESwish module.
         *
         * Expects an input initializer list containing a tensor as its primary argument.
         *
         * @param tensors Initializer list containing inputs (wrapped in `std::any`).
         * @return std::any Output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
    };
}