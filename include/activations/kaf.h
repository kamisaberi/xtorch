#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the KAF (Kernel Activation Function) and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the Kernel Activation Function (KAF) on an input tensor.
     *
     * KAF models activation functions non-parametrically as 1D linear combinations
     * of kernel functions (such as Gaussian/RBF kernels) evaluated over a set of dictionary points.
     *
     * @param x The input tensor.
     * @param dictionary_coefs Tensor containing kernel expansion coefficients.
     * @param boundary_params Tensor containing dictionary boundary/grid parameters.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor kaf(const torch::Tensor& x,
                      const torch::Tensor& dictionary_coefs, // Shape [D] or [1, D] for broadcasting
                      const torch::Tensor& boundary_params // Shape [D-1] or [1, D-1] for broadcasting
    );

    /**
     * @brief A module wrapper for the Kernel Activation Function (KAF).
     *
     * Inherits from `xt::Module` to enable dynamic execution within neural network models.
     */
    struct KAF : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the KAF module.
         */
        KAF() = default;

        /**
         * @brief Performs the forward pass for the KAF module.
         *
         * @param tensors An initializer list of `std::any` expected to contain the input tensor,
         *                dictionary coefficients, and boundary parameters.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}