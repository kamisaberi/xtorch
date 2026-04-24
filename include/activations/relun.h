#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the ReLUN (Rectified Linear Unit N) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the ReLUN (Rectified Linear Unit N) activation function on an input tensor.
     *
     * ReLUN applies a rectified linear unit capped at a maximum upper bound threshold \p n_val:
     * \f$ \text{ReLUN}(x) = \min(\max(0, x), n\_val) \f$
     *
     * @param x The input tensor.
     * @param n_val Upper saturation threshold value (defaults to 1.0).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor relun(const torch::Tensor& x, double n_val = 1.0);

    /**
     * @brief A module wrapper for the ReLUN activation function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network architectures.
     */
    struct ReLUN : xt::Module {
    public:
        /**
         * @brief Default constructor for the ReLUN module.
         */
        ReLUN() = default;

        /**
         * @brief Performs the forward pass for the ReLUN module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}