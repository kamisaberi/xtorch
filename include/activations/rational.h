#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Rational activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes a rational activation function on an input tensor.
     *
     * Evaluates a rational function $R(x) = \frac{P(x)}{|Q(x)| + \epsilon}$ where $P(x)$
     * and $Q(x)$ are polynomials parameterized by \p P_coeffs and \p Q_coeffs:
     * \f$ R(x) = \frac{\sum_{i=0}^{m} p_i x^i}{\left|\sum_{j=1}^{n} q_j x^j + 1\right| + \epsilon} \f$
     *
     * @param x The input tensor to transform.
     * @param P_coeffs Numerator polynomial coefficients tensor.
     * @param Q_coeffs Denominator polynomial coefficients tensor.
     * @param epsilon Small constant for denominator numerical stability (defaults to 1e-7).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor rational(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, ..., q_1] (q_0 is fixed to 1)
        double epsilon = 1e-7 // For denominator stability
    );


    /**
     * @brief A module wrapper for rational activation functions.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct Rational : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the Rational module.
         */
        Rational() = default;

        /**
         * @brief Performs the forward pass for the Rational module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x`, `P_coeffs`, and `Q_coeffs`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}