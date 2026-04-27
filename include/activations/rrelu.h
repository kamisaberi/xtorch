#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the RReLU (Randomized Leaky Rectified Linear Unit) activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the RReLU (Randomized Leaky Rectified Linear Unit) activation on an input tensor.
     *
     * RReLU applies a leaky ReLU activation where negative input values are scaled by a factor $a$.
     * During training (\p training = true), $a$ is randomly drawn from a uniform distribution
     * $\mathcal{U}(\text{lower}, \text{upper})$. During evaluation (\p training = false), $a$ is fixed
     * to the mean $\frac{\text{lower} + \text{upper}}{2}$:
     * \f$ \text{RReLU}(x) = \max(0, x) + a \cdot \min(0, x) \f$
     *
     * @param x The input tensor.
     * @param lower Lower bound of the uniform distribution for negative slope (defaults to 1.0 / 8.0).
     * @param upper Upper bound of the uniform distribution for negative slope (defaults to 1.0 / 3.0).
     * @param training If true, samples random slope $a$; if false, uses fixed mean slope (defaults to false).
     * @param generator Optional random number generator for reproducible sampling (defaults to c10::nullopt).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor rrelu(
        const torch::Tensor& x,
        double lower = 1.0 / 8.0,
        double upper = 1.0 / 3.0,
        bool training = false,
        c10::optional<at::Generator> generator = c10::nullopt
    );


    /**
     * @brief A module wrapper for the Randomized Leaky ReLU (RReLU) activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct RReLU : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the RReLU module.
         */
        RReLU() = default;

        /**
         * @brief Performs the forward pass for the RReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}