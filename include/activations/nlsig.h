#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the NLSIG (Non-Linear Sigmoid) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the NLSIG (Non-Linear Sigmoid) activation function on an input tensor.
     *
     * Applies a parameterized non-linear sigmoid transformation to the input tensor \p x
     * controlled by shape and scaling hyperparameters \p a and \p b.
     *
     * @param x The input tensor.
     * @param a Scaling or slope parameter (defaults to 1.0).
     * @param b Shift or curvature parameter (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor nlsig(const torch::Tensor& x, double a = 1.0, double b = 1.0);

    /**
     * @brief A module wrapper for the NLSIG activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct NLSIG : xt::Module {
    public:
        /**
         * @brief Default constructor for the NLSIG module.
         */
        NLSIG() = default;

        /**
         * @brief Performs the forward pass for the NLSIG module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}