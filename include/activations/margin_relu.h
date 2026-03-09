#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the MarginReLU activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the MarginReLU activation function on an input tensor.
     *
     * MarginReLU applies a rectified linear activation bounded or parameterized
     * by negative (\p margin_neg) and positive (\p margin_pos) margin thresholds.
     *
     * @param x The input tensor to transform.
     * @param margin_neg Negative margin threshold parameter (defaults to 0.1).
     * @param margin_pos Positive margin threshold parameter (defaults to 0.9).
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor margin_relu(const torch::Tensor& x, double margin_neg = 0.1, double margin_pos = 0.9) ;

    /**
     * @brief A module wrapper for the MarginReLU activation function.
     *
     * Inherits from `xt::Module` to enable dynamic execution within neural network models.
     */
    struct MarginReLU : xt::Module {
    public:
        /**
         * @brief Default constructor for the MarginReLU module.
         */
        MarginReLU() = default;

        /**
         * @brief Performs the forward pass for the MarginReLU module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}