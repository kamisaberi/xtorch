/**
 * @file evonorm_s0.h
 * @brief Declaration of the EvoNorm-S0 activation-normalization function and its corresponding xt::Module wrapper.
 */

#pragma once

#include "common.h"

/**
 * @namespace xt::activations
 * @brief Namespace containing extended activation functions and modules for xTorch.
 */
namespace xt::activations
{
    /**
     * @brief Computes the EvoNorm-S0 (Evolving Normalization - Sample-based S0) activation function.
     *
     * Applies sample-based evolving normalization combining group normalization and activation.
     *
     * @param x Input tensor to process.
     * @param gamma Scale parameter tensor.
     * @param beta Bias parameter tensor.
     * @param v_param Value parameter tensor for non-linear component weighting.
     * @param num_groups Number of groups to use for feature grouping.
     * @param eps Small epsilon value for numerical stability (default: 1e-5).
     * @return torch::Tensor Processed output tensor.
     */
    torch::Tensor evonorm_s0(const torch::Tensor& x, const torch::Tensor& gamma, const torch::Tensor& beta,
                             const torch::Tensor& v_param, int64_t num_groups,
                             double eps = 1e-5);

    /**
     * @struct EvonormS0
     * @brief High-level module wrapper for the EvoNorm-S0 function.
     *
     * Inherits from `xt::Module` to enable dynamic forward invocation in xTorch pipelines.
     */
    struct EvonormS0 : xt::Module
    {
    public:
        /**
         * @brief Default constructor for EvonormS0.
         */
        EvonormS0() = default;

        /**
         * @brief Forward pass for the EvonormS0 module.
         *
         * Expects an input initializer list containing tensors and arguments wrapped in `std::any`.
         *
         * @param tensors Initializer list containing inputs (wrapped in `std::any`).
         * @return std::any Output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}