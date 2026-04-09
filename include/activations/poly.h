#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the polynomial activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes a polynomial activation function on an input tensor.
     *
     * Evaluates a polynomial function on input tensor \p x using coefficient values in \p coefficients:
     * \f$ P(x) = c_d x^d + c_{d-1} x^{d-1} + \dots + c_1 x + c_0 \f$
     *
     * @param x The input tensor to transform.
     * @param coefficients Tensor containing polynomial coefficients ordered from degree \p d down to 0.
     * @return torch::Tensor The evaluated polynomial output tensor.
     */
    torch::Tensor poly(
        const torch::Tensor& x,
        const torch::Tensor& coefficients // Shape [degree, degree-1, ..., 1, 0]
    );


    /**
     * @brief A module wrapper for the polynomial activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Poly : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the Poly module.
         */
        Poly() = default;

        /**
         * @brief Performs the forward pass for the Poly module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x` and polynomial coefficients.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}