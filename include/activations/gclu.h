/**
* @file gclu.h
 * @brief Declaration of the GCLU (Gated Circular/Channel Linear Unit) activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the GCLU (Gated Circular/Channel Linear Unit) activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param dim Dimension along which the tensor is split for gating (default: 1).
     * @return torch::Tensor Output tensor with the GCLU activation applied.
     */
    torch::Tensor gclu(torch::Tensor x, int64_t dim = 1);

    /**
     * @struct GCLU
     * @brief High-level module wrapper for the GCLU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct GCLU : xt::Module {
    public:
        /**
         * @brief Default constructor for GCLU.
         */
        GCLU() = default;

        /**
         * @brief Forward pass for the GCLU module.
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