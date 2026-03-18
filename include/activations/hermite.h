#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the Hermite polynomial activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the Hermite polynomial-based activation function on a tensor.
     *
     * Applies a Hermite polynomial transformation to the elements of the input tensor.
     *
     * @param x Reference to the input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor hermite(torch::Tensor& x);

    /**
     * @brief A module wrapper for the Hermite activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network models.
     */
    struct Hermite : xt::Module {
    public:
        /**
         * @brief Default constructor for the Hermite module.
         */
        Hermite() = default;

        /**
         * @brief Performs the forward pass for the Hermite module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}