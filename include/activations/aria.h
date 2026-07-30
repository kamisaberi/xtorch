/**
 * @file aria.h
 * @brief Declaration of the ARiA activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the ARiA activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param alpha Parameter controlling the shape or scaling of the activation (default: 1.0).
     * @param beta Parameter controlling the scale or behavior of the function (default: 1.0).
     * @return torch::Tensor Output tensor with the ARiA activation applied.
     */
    torch::Tensor aria(torch::Tensor x, double alpha = 1.0, double beta = 1.0);

    /**
     * @struct ARiA
     * @brief High-level module wrapper for the ARiA activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct ARiA : xt::Module {
    public:
        /**
         * @brief Default constructor for ARiA.
         */
        ARiA() = default;

        /**
         * @brief Forward pass for the ARiA module.
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