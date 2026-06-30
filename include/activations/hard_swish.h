#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the HardSwish activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the HardSwish activation function on an input tensor.
     *
     * HardSwish is a computationally efficient piece-wise linear approximation
     * of the Swish activation function, defined as:
     * \f$ \text{HardSwish}(x) = x \cdot \frac{\text{ReLU6}(x + 3)}{6} \f$
     *
     * @param x The input tensor.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor hard_swich(torch::Tensor x);

    /**
     * @brief A module wrapper for the HardSwish activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network models.
     */
    struct HardSwish : xt::Module {
    public:
        /**
         * @brief Default constructor for the HardSwish module.
         */
        HardSwish() = default;

        /**
         * @brief Performs the forward pass for the HardSwish module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}