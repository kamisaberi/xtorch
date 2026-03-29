#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the NFN (Non-linear Filter Network / Neural Functional Network) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the NFN activation/filter operation on an input tensor.
     *
     * Applies parameterized non-linear filter transformations controlled by weight tensors
     * \p alpha and \p beta across input and output filter channels.
     *
     * @param x The input tensor.
     * @param alpha Tensor containing primary filter coefficients.
     * @param beta Tensor containing secondary/bias filter coefficients.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor nfn(
        const torch::Tensor& x,
        const torch::Tensor& alpha, // Shape (num_filters_out, num_filters_in, filter_size)
        const torch::Tensor& beta // Shape (num_filters_out, num_filters_in, filter_size)
    );

    /**
     * @brief A module wrapper for the NFN activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct NFN : xt::Module {
    public:
        /**
         * @brief Default constructor for the NFN module.
         */
        NFN() = default;

        /**
         * @brief Performs the forward pass for the NFN module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensor `x`, `alpha`, and `beta`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}