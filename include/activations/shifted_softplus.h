#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ShiftedSoftplus activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Constant representing the natural logarithm of 2 (\f$\ln(2)\f$).
     */
    const double LN_2 = std::log(2.0); // More portable way to get log(2)

    /**
     * @brief Computes the Shifted Softplus activation function on an input tensor.
     *
     * Shifted Softplus subtracts a shift value \p shift_val from the Softplus function,
     * ensuring that $f(0) = 0$ when \p shift_val is set to \f$\ln(2)\f$:
     * \f$ \text{ShiftedSoftplus}(x) = \ln(1 + e^x) - \text{shift\_val} \f$
     *
     * @param x The input tensor.
     * @param shift_val The constant to subtract from Softplus (defaults to LN_2 = ln(2)).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor shifted_softplus(const torch::Tensor& x, double shift_val = LN_2);

    /**
     * @brief A module wrapper for the ShiftedSoftplus activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct ShiftedSoftplus : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ShiftedSoftplus module.
         */
        ShiftedSoftplus() = default;

        /**
         * @brief Performs the forward pass for the ShiftedSoftplus module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}