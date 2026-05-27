#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SERLU (Scaled Exponential Relative Linear Unit) activation function and module.
 */

namespace xt::activations
{
    /**
     * @brief Computes the SERLU (Scaled Exponential Relative Linear Unit) activation on an input tensor.
     *
     * SERLU is a self-normalizing activation function defined as:
     * \f$ \text{SERLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \lambda \cdot \alpha \cdot x \cdot e^x & \text{if } x \le 0 \end{cases} \f$
     *
     * @param x The input tensor.
     * @param lambda_serlu Scaling hyperparameter (defaults to 1.0507).
     * @param alpha_serlu Saturation shape hyperparameter (defaults to 1.67326).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor serlu(const torch::Tensor& x, double lambda_serlu = 1.0507, double alpha_serlu = 1.67326);

    /**
     * @brief A module wrapper for the SERLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SERLU : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the SERLU module.
         */
        SERLU() = default;

        /**
         * @brief Performs the forward pass for the SERLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}