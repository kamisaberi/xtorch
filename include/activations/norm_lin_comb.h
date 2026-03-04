#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the normalized linear combination activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes a normalized linear combination of base activation functions on an input tensor.
     *
     * Evaluates each callable in \p base_functions on \p x, combines the outputs using \p coefficients,
     * and applies normalization with \p eps for numerical stability.
     *
     * @param x The input tensor.
     * @param base_functions Vector of callable functions taking a tensor reference and returning a tensor.
     * @param coefficients Tensor of combining weights (Shape: num_base_functions).
     * @param eps Small constant added for numerical stability during normalization (defaults to 1e-5).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor norm_lin_comb(
        const torch::Tensor& x,
        const std::vector<std::function<torch::Tensor(const torch::Tensor&)>>& base_functions,
        const torch::Tensor& coefficients, // Shape (num_base_functions)
        double eps = 1e-5
    );


    /**
     * @brief A module wrapper for normalized linear combination activations.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct NormLinComb : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the NormLinComb module.
         */
        NormLinComb() = default;

        /**
         * @brief Performs the forward pass for the NormLinComb module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}