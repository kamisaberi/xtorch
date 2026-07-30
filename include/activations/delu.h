/**
* @file delu.h
 * @brief Declaration of the DELU activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations
{
    /**
     * @brief Computes the DELU activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param alpha Parameter controlling the shape or scaling aspect of the activation (default: 1.0).
     * @param gamma Parameter controlling the exponential or scale factor (default: 1.0).
     * @return torch::Tensor Output tensor with the DELU activation applied.
     */
    torch::Tensor delu(torch::Tensor x, double alpha = 1.0, double gamma = 1.0);

    /**
     * @struct DELU
     * @brief High-level module wrapper for the DELU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct DELU : xt::Module
    {
    public:
        /**
         * @brief Default constructor for DELU.
         */
        DELU() = default;

        /**
         * @brief Forward pass for the DELU module.
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