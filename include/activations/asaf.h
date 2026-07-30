/**
 * @file asaf.h
 * @brief Declaration of the ASAF activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the ASAF activation function on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param p_param Parameter controlling the shape or steepness of the activation (default: 1.0).
     * @param q_param Parameter controlling the secondary shape or scale aspect (default: 1.0).
     * @return torch::Tensor Output tensor with the ASAF activation applied.
     */
    torch::Tensor asaf(torch::Tensor x, double p_param = 1.0, double q_param = 1.0);

    /**
     * @struct ASAF
     * @brief High-level module wrapper for the ASAF activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct ASAF : xt::Module {
    public:
        /**
         * @brief Default constructor for ASAF.
         */
        ASAF() = default;

        /**
         * @brief Forward pass for the ASAF module.
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