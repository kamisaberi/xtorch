/**
 * @file colu.h
 * @brief Declaration of the CoLU (Collapsing Linear Unit) activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the CoLU (Collapsing Linear Unit) activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param M_val Parameter controlling the bound or collapse threshold (default: 1.0).
     * @return torch::Tensor Output tensor with the CoLU activation applied.
     */
    torch::Tensor colu(torch::Tensor x, double M_val = 1.0);

    /**
     * @struct CoLU
     * @brief High-level module wrapper for the CoLU activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct CoLU : xt::Module {
    public:
        /**
         * @brief Default constructor for CoLU.
         */
        CoLU() = default;

        /**
         * @brief Forward pass for the CoLU module.
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