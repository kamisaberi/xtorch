#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the GoLU (Gompertz Linear Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the GoLU (Gompertz Linear Unit) activation on an input tensor.
     *
     * GoLU is an asymmetric self-gated activation function that utilizes the Gompertz function
     * as its gating mechanism to stabilize training and reduce output variance in latent space.
     *
     * @param x The input tensor to activate.
     * @param alpha Scaling/shape parameter for the GoLU activation (defaults to 1.0).
     * @param dim The dimension along which operation or splitting occurs (defaults to 1).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor golu(torch::Tensor x, double alpha = 1.0, int64_t dim = 1);

    /**
     * @brief A module wrapper for the GoLU activation function.
     *
     * Inherits from `xt::Module` to enable usage within neural network pipeline structures.
     */
    struct GoLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the GoLU module.
         */
        GoLU() = default;

        /**
         * @brief Executes the forward pass for the GoLU module.
         *
         * @param tensors An initializer list of `std::any` elements containing the input tensor
         *                and optional parameters for the forward execution.
         * @return std::any The activated output wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}