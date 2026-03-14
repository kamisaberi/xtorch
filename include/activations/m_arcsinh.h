#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the MArcsinh (Modified Inverse Hyperbolic Sine) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Modified Arcsinh (MArcsinh) activation function on an input tensor.
     *
     * Applies a scaled inverse hyperbolic sine function parameterized by \p m:
     * \f$ f(x) = m \cdot \text{arcsinh}(x) \f$
     *
     * @param x The input tensor.
     * @param m Scaling hyperparameter (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor m_arcsinh(const torch::Tensor& x, double m = 1.0) ;

    /**
     * @brief A module wrapper for the MArcsinh activation function.
     *
     * Inherits from `xt::Module` to enable dynamic execution within neural network models.
     */
    struct MArcsinh : xt::Module {
    public:
        /**
         * @brief Default constructor for the MArcsinh module.
         */
        MArcsinh() = default;

        /**
         * @brief Performs the forward pass for the MArcsinh module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}