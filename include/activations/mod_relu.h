#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ModReLU (Modulus ReLU) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the ModReLU (Modulus ReLU) activation function on an input tensor.
     *
     * ModReLU scales magnitude based on a bias parameter while preserving phase/direction:
     * \f$ \text{modReLU}(x) = \text{ReLU}(|x| + b) \frac{x}{|x|} \f$
     *
     * @param x The input tensor (real or complex).
     * @param b Bias tensor controlling magnitude thresholding.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor mod_relu(const torch::Tensor& x, const torch::Tensor& b);

    /**
     * @brief A module wrapper for the ModReLU activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct ModReLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the ModReLU module.
         */
        ModReLU() = default;

        /**
         * @brief Performs the forward pass for the ModReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x` and bias tensor `b`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}