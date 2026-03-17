#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for Kolmogorov-Arnold Network (KAN) B-spline activation functions and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes B-spline basis functions for an input tensor given a knot grid.
     *
     * Calculates basis function values (e.g., via Cox-de Boor recursion formula)
     * for order \p k_order evaluated at points in \p x along the specified \p grid.
     *
     * @param x The input tensor containing evaluation points.
     * @param grid The knot grid tensor.
     * @param k_order The B-spline order (e.g., 3 for quadratic, 4 for cubic).
     * @return torch::Tensor Tensor containing evaluated B-spline basis function values.
     */
    torch::Tensor b_spline_basis(const torch::Tensor& x, const torch::Tensor& grid, int k_order);

    /**
     * @brief Computes the Kolmogorov-Arnold Network (KAN) spline activation.
     *
     * Combines a base activation function scaled by \p base_activation_weight
     * with a linear combination of B-spline basis functions parameterised by \p spline_weights.
     *
     * @param x The input tensor.
     * @param spline_weights Spline coefficient weights.
     * @param grid_internal Knot grid points bounding interval ranges.
     * @param k_order Spline order (e.g., 4 for cubic splines).
     * @param base_activation_weight Scale factor for the base activation function.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor kan_spline_activation(
        const torch::Tensor& x,
        const torch::Tensor& spline_weights, // Shape (G + k - 1)
        const torch::Tensor& grid_internal, // Shape (G + 1) for G intervals
        int k_order, // e.g., 4 for cubic
        double base_activation_weight // w
    );


    /**
     * @brief A module wrapper for Kolmogorov-Arnold Network (KAN) activations.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct KAN : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the KAN module.
         */
        KAN() = default;

        /**
         * @brief Performs the forward pass for the KAN module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor, spline weights, and grid parameters.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}