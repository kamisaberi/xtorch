#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the HardELiSH activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the HardELiSH (Hard Exponential Linear Sigmoid Unit) activation function on a tensor.
     *
     * HardELiSH is a computationally efficient variant of the ELiSH activation function.
     * It combines a HardSigmoid with a linear function for positive inputs and a HardSigmoid
     * with ELU (Exponential Linear Unit) for negative inputs.
     *
     * @param x Reference to the input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor hard_elish(torch::Tensor& x);

    /**
     * @brief A module wrapper for the HardELiSH activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network models.
     */
    struct HardELiSH : xt::Module {
    public:
        /**
         * @brief Default constructor for the HardELiSH module.
         */
        HardELiSH() = default;

        /**
         * @brief Performs the forward pass for the HardELiSH module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}