#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the PAU (Padé Activation Unit) rational activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the Padé Activation Unit (PAU) rational activation function on an input tensor.
     *
     * PAU is a learnable rational activation function based on Padé approximants, defined as
     * the ratio of two polynomials $P(x)$ and $Q(x)$:
     * \f$ \text{PAU}(x) = \frac{P(x)}{|Q(x)| + \epsilon} \f$
     *
     * @param x The input tensor.
     * @param P_coeffs Numerator polynomial coefficients tensor.
     * @param Q_coeffs Denominator polynomial coefficients tensor.
     * @param epsilon Small constant for numerical stability to prevent division by zero (defaults to 1e-7).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor pau(
        const torch::Tensor& x,
        const torch::Tensor& P_coeffs, // Numerator coefficients [p_m, p_{m-1}, ..., p_1, p_0]
        const torch::Tensor& Q_coeffs, // Denominator coefficients [q_n, q_{n-1}, ..., q_1, 1.0] (q_0 is fixed to 1)
        double epsilon = 1e-7
    );


    /**
     * @brief A module wrapper for the Padé Activation Unit (PAU).
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network models.
     */
    struct PAU : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the PAU module.
         */
        PAU() = default;

        /**
         * @brief Performs the forward pass for the PAU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x`, `P_coeffs`, and `Q_coeffs`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}