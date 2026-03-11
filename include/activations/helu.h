#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the HeLU (He Exponential Linear Unit) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the HeLU activation function on an input tensor.
     *
     * HeLU is a parameterized variant of the Exponential Linear Unit (ELU) modified
     * with an explicit scale parameter (\p lambda_param) and exponential scaling (\p alpha).
     *
     * @param x The input tensor.
     * @param alpha Parameter controlling the saturation value for negative inputs (defaults to 1.0).
     * @param lambda_param Scale hyperparameter applied to the output activation (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor helu(torch::Tensor x, double alpha = 1.0, double lambda_param = 1.0);

    /**
     * @brief A module wrapper for the HeLU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic invocation within neural network layers.
     */
    struct HeLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the HeLU module.
         */
        HeLU() = default;

        /**
         * @brief Performs the forward pass for the HeLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}