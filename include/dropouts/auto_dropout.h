#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the AutoDropout function and module.
 */

namespace xt::dropouts
{
    /**
     * @brief Applies auto-dropout to an input tensor.
     *
     * AutoDropout scales or zero-masks input elements using automatically adjusted or learnable
     * dropout probability distributions.
     *
     * @param x The input tensor.
     * @return torch::Tensor The output tensor with auto-dropout applied.
     */
    torch::Tensor auto_dropout(torch::Tensor x);

    /**
     * @brief A module wrapper for AutoDropout with learnable dropout rates.
     *
     * Inherits from `xt::Module` and maintains a learnable log-alpha parameter tensor (`log_alpha_`)
     * to automatically tune feature dropout rates during model training.
     */
    struct AutoDropout : xt::Module
    {
    public:
        /**
         * @brief Constructs an AutoDropout module.
         *
         * @param probability_shape Shape of the learnable dropout probability parameter tensor (defaults to empty `{}`).
         * @param initial_dropout_rate Initial dropout probability value (defaults to 0.05).
         */
        explicit AutoDropout(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05);

        /**
         * @brief Performs the forward pass for the AutoDropout module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Learnable parameter tensor representing log-alpha (log-variance ratio) for automated dropout rates.
         */
        torch::Tensor log_alpha_;
    };
}