/**
 * @file asu.h
 * @brief Declaration of the ASU activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the ASU activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param alpha Parameter controlling the primary scaling or shape (default: 1.0).
     * @param beta Parameter controlling the secondary shape or scale (default: 1.0).
     * @param gamma Parameter controlling the shift or offset (default: 0.0).
     * @return torch::Tensor Output tensor with the ASU activation applied.
     */
    torch::Tensor asu(torch::Tensor x, double alpha = 1.0, double beta = 1.0, double gamma = 0.0);

    /**
     * @struct ASU
     * @brief High-level module wrapper for the ASU activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct ASU : xt::Module {
    public:
        /**
         * @brief Default constructor for ASU.
         */
        ASU() = default;

        /**
         * @brief Forward pass for the ASU module.
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