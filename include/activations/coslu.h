/**
 * @file coslu.h
 * @brief Declaration of the CosLU (Cosine Linear Unit) activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the CosLU (Cosine Linear Unit) activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @return torch::Tensor Output tensor with the CosLU activation applied.
     */
    torch::Tensor coslu(torch::Tensor x);

    /**
     * @struct CosLU
     * @brief High-level module wrapper for the CosLU activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct CosLU : xt::Module {
    public:
        /**
         * @brief Default constructor for CosLU.
         */
        CosLU() = default;

        /**
         * @brief Forward pass for the CosLU module.
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