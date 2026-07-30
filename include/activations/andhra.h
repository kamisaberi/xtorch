/**
 * @file andhra.h
 * @brief Declaration of the ANDHRA activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the ANDHRA activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param alpha Scaling or shape parameter (default: 1.0).
     * @param beta Slope or offset parameter (default: 0.01).
     * @return torch::Tensor Output tensor with the ANDHRA activation applied.
     */
    torch::Tensor andhra(torch::Tensor x, double alpha = 1.0, double beta = 0.01);

    /**
     * @struct ANDHRA
     * @brief High-level module wrapper for the ANDHRA activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct ANDHRA : xt::Module {
    public:
        /**
         * @brief Default constructor for ANDHRA.
         */
        ANDHRA() = default;

        /**
         * @brief Forward pass for the ANDHRA module.
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