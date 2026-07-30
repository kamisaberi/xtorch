/**
 * @file crelu.h
 * @brief Declaration of the CReLU (Concatenated Rectified Linear Unit) activation function and its corresponding xt::Module wrapper.
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
     * @brief Computes the CReLU (Concatenated Rectified Linear Unit) activation function on an input tensor.
     * 
     * Concatenates ReLU(x) and ReLU(-x) along the specified dimension.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param dim Dimension along which to concatenate positive and negative activations (default: 1).
     * @return torch::Tensor Output tensor with doubled size along the specified dimension.
     */
    torch::Tensor crelu(torch::Tensor x, int64_t dim = 1);

    /**
     * @struct CReLU
     * @brief High-level module wrapper for the CReLU activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct CReLU : xt::Module
    {
    public:
        /**
         * @brief Default constructor for CReLU.
         */
        CReLU() = default;

        /**
         * @brief Forward pass for the CReLU module.
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