/**
 * @file ahaf.h
 * @brief Declaration of the AHAF (Adaptive Hyperbolic Activation Function) and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
    /**
     * @brief Computes the AHAF (Adaptive Hyperbolic Activation Function) on an input tensor.
     * 
     * @param x Input tensor to apply the activation function on.
     * @param p_param Shape parameter controlling the behavior or steepness of the function (default: 1.0).
     * @return torch::Tensor Output tensor with the AHAF activation applied.
     */
    torch::Tensor ahaf(torch::Tensor x, double p_param = 1.0);

    /**
     * @struct AHAF
     * @brief High-level module wrapper for the AHAF activation function.
     * 
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct AHAF : xt::Module {
    public:
        /**
         * @brief Default constructor for AHAF.
         */
        AHAF() = default;

        /**
         * @brief Forward pass for the AHAF module.
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