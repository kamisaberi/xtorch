#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the linear combination activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes a linear combination of multiple base activation functions applied to an input tensor.
     *
     * Evaluates each function in \p base_functions on \p x, multiplies the result by its
     * corresponding scalar in \p coefficients, and sums the weighted outputs:
     * \f$ f(x) = \sum_{i=1}^{N} c_i \cdot \phi_i(x) \f$
     *
     * @param x The input tensor to transform.
     * @param base_functions Vector of callable functions taking a tensor reference and returning a tensor.
     * @param coefficients Weights tensor matching the number of base functions (Shape: num_base_functions).
     * @return torch::Tensor The resulting weighted linear combination tensor.
     */
    torch::Tensor lin_comb(
        const torch::Tensor& x,
        const std::vector<std::function<torch::Tensor(const torch::Tensor&)>>& base_functions,
        const torch::Tensor& coefficients // Shape (num_base_functions)
    );

    /**
     * @brief A module wrapper for linear combinations of base activation functions.
     *
     * Inherits from `xt::Module` to support execution within neural network models.
     */
    struct LinComb : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the LinComb module.
         */
        LinComb() = default;

        /**
         * @brief Performs the forward pass for the LinComb module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                passed to the module execution.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}