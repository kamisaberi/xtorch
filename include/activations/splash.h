#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SPLASH activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SPLASH activation function on an input tensor.
     *
     * Applies the parameterized SPLASH activation function to the input tensor \p x,
     * parameterized by scaling, ratio/range, and bias parameters \p S, \p R, and \p B.
     *
     * @param x The input tensor.
     * @param S Scale/slope parameter (defaults to 1.0).
     * @param R Ratio/range parameter (defaults to 0.5).
     * @param B Bias/base parameter (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor splash(const torch::Tensor& x, double S = 1.0, double R = 0.5, double B = 1.0);

    /**
     * @brief A module wrapper for the SPLASH activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SPLASH : xt::Module {
    public:
        /**
         * @brief Default constructor for the SPLASH module.
         */
        SPLASH() = default;

        /**
         * @brief Performs the forward pass for the SPLASH module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}