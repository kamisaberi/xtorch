#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the SquaredReLU activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the SquaredReLU activation function on an input tensor.
     *
     * SquaredReLU applies the square of the Rectified Linear Unit (ReLU) function:
     * \f$ \text{SquaredReLU}(x) = \max(0, x)^2 \f$
     *
     * @param x The input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor squared_relu(const torch::Tensor& x);

    /**
     * @brief A module wrapper for the SquaredReLU activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct SquaredReLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the SquaredReLU module.
         */
        SquaredReLU() = default;

        /**
         * @brief Performs the forward pass for the SquaredReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}