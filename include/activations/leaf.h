#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the LEAF (Learnable Activation Function) and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the LEAF (Learnable Activation Function) on an input tensor.
     *
     * LEAF evaluates a parameterized, learnable activation function parameterized
     * by component scale, rotation/ratio, and shift weights.
     *
     * @param x The input tensor to process.
     * @param s_weights Tensor containing scale parameter weights.
     * @param r_weights Tensor containing rotation/ratio parameter weights.
     * @param u_weights Tensor containing shift/univariate parameter weights.
     * @return torch::Tensor The activated output tensor.
     */
    torch::Tensor leaf(const torch::Tensor& x,
                       const torch::Tensor& s_weights, // Shape (L)
                       const torch::Tensor& r_weights, // Shape (L)
                       const torch::Tensor& u_weights // Shape (L)
    );

    /**
     * @brief A module wrapper for the LEAF activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct LEAF : xt::Module {
    public:
        /**
         * @brief Default constructor for the LEAF module.
         */
        LEAF() = default;

        /**
         * @brief Performs the forward pass for the LEAF module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                including input tensor `x`, `s_weights`, `r_weights`, and `u_weights`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}