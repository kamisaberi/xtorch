/**
 * @file aglu.h
 * @brief Declaration of the AGLU (Adaptive Gated Linear Unit) activation function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations {
/**
     * @brief Computes the AGLU (Adaptive Gated Linear Unit) activation function on an input tensor.
     *
     * @param x Input tensor to apply the activation function on.
     * @param s Scale or shape parameter controlling the gating behavior (default: 1.0).
     * @return torch::Tensor Output tensor with the AGLU activation applied.
     */
torch::Tensor aglu(torch::Tensor x, double s = 1.0);

/**
     * @struct AGLU
     * @brief High-level module wrapper for the AGLU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
struct AGLU : xt::Module {
public:
    /**
         * @brief Default constructor for AGLU.
         */
    AGLU() = default;

    /**
         * @brief Forward pass for the AGLU module.
         *
         * Expects an input initializer list containing a tensor as its primary argument.
         *
         * @param tensors Initializer list containing inputs (wrapped in `std::any`).
         * @return std::any Output tensor wrapped in `std::any`.
         */
    auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

};
}
