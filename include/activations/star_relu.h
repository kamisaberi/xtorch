#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the StarReLU activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the StarReLU activation function on an input tensor.
     *
     * StarReLU applies a scaled, squared/leaky rectified linear transformation with a bias offset:
     * \f$ \text{StarReLU}(x) = \text{scale} \cdot (\text{ReLU}(x))^2 + \text{bias} \f$
     * (generalized with \p relu_slope for positive inputs and \p leaky_slope for negative inputs).
     *
     * @param x The input tensor.
     * @param scale Scaling factor parameter applied to the output (defaults to 1.0).
     * @param bias Bias offset parameter added to the output (defaults to 0.0).
     * @param relu_slope Slope factor for the positive region (defaults to 1.0).
     * @param leaky_slope Slope factor for the negative region (defaults to 0.01).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor star_relu(const torch::Tensor& x, double scale = 1.0, double bias = 0.0, double relu_slope = 1.0,
                        double leaky_slope = 0.01);


    /**
     * @brief A module wrapper for the StarReLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct StarReLU: xt::Module {
    public:
        /**
         * @brief Default constructor for the StarReLU module.
         */
        StarReLU() = default;

        /**
         * @brief Performs the forward pass for the StarReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}