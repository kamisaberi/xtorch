/**
* @file dra.h
 * @brief Declaration of the DRA activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the DRA activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param alpha Parameter controlling the slope or scaling factor (default: 1.0).
     * @return torch::Tensor Output tensor with the DRA activation applied.
     */
    torch::Tensor dra(torch::Tensor x, double alpha = 1.0);

    /**
     * @struct DRA
     * @brief High-level module wrapper for the DRA activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct DRA : xt::Module {
    public:
        /**
         * @brief Default constructor for DRA.
         */
        DRA() = default;

        /**
         * @brief Forward pass for the DRA module.
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