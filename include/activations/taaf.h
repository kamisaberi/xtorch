#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the TAAF (Trainable Adaptive Activation Function) and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the TAAF (Trainable Adaptive Activation Function) on an input tensor.
     *
     * TAAF evaluates an adaptive activation function parameterized by a per-channel
     * learnable parameter \p alpha and a global scaling hyperparameter \p beta.
     *
     * @param x The input tensor to transform.
     * @param alpha Per-channel learnable parameter tensor.
     * @param beta Global hyperparameter or fixed scale factor (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor taaf(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Per-channel learnable parameter
        double beta = 1.0 // Global hyperparameter or fixed
    );


    /**
     * @brief A module wrapper for the TAAF activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct TAAF : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the TAAF module.
         */
        TAAF() = default;

        /**
         * @brief Performs the forward pass for the TAAF module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x` and parameter `alpha`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}