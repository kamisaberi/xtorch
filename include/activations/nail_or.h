#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the NailOr (Logit-space Logical OR) activation function and module.
 */

namespace xt::activations {
    /**
     * @brief Computes the NailOr (Logit-space Probabilistic OR) activation function.
     *
     * NailOr computes the logit-space equivalent of the probabilistic Boolean OR operator
     * between two logit input tensors \p x and \p z (as introduced in "Logical Activation
     * Functions: Logit-space equivalents of Probabilistic Boolean Operators", Lowe et al.).
     *
     * @param x The first input tensor (logits).
     * @param z The second input tensor (logits).
     * @return torch::Tensor The output tensor resulting from the logit-space logical OR operation.
     */
    torch::Tensor nail_or(const torch::Tensor& x, const torch::Tensor& z);

    /**
     * @brief A module wrapper for the NailOr activation function.
     *
     * Inherits from `xt::Module` to support execution within neural network architectures.
     */
    struct NailOr : xt::Module {
    public:
        /**
         * @brief Default constructor for the NailOr module.
         */
        NailOr() = default;

        /**
         * @brief Performs the forward pass for the NailOr module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors
         *                such as input tensors `x` and `z`.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}