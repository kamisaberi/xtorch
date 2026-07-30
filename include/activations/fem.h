/**
* @file fem.h
 * @brief Declaration of the FEM activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the FEM activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param alpha Primary shape or scaling parameter (default: 0.25).
     * @param beta Secondary shape or slope parameter (default: 0.01).
     * @return torch::Tensor Output tensor with the FEM activation applied.
     */
    torch::Tensor fem(torch::Tensor x, double alpha = 0.25, double beta = 0.01);

    /**
     * @struct FEM
     * @brief High-level module wrapper for the FEM activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct FEM : xt::Module {
    public:
        /**
         * @brief Default constructor for FEM.
         */
        FEM() = default;

        /**
         * @brief Forward pass for the FEM module.
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