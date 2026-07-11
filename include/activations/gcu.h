/**
* @file gcu.h
 * @brief Declaration of the GCU (Growing Cosine Unit) activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the GCU (Growing Cosine Unit) activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @return torch::Tensor Output tensor with the GCU activation applied.
     */
    torch::Tensor gcu(torch::Tensor x);

    /**
     * @struct GCU
     * @brief High-level module wrapper for the GCU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct GCU : xt::Module {
    public:
        /**
         * @brief Default constructor for GCU.
         */
        GCU() = default;

        /**
         * @brief Forward pass for the GCU module.
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