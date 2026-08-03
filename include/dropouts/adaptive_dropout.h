#pragma once

#include "common.h"
#include <torch/torch.h>
#include <vector>
#include <cmath>     // For std::log
#include <ostream>   // For std::ostream


/**
 * @file
 * @brief Header file for the Adaptive Dropout function and module.
 */

namespace xt::dropouts
{
    /**
     * @brief Applies adaptive dropout to an input tensor.
     *
     * Adaptive Dropout scales or masks input elements using learnable or parameter-dependent
     * dropout probabilities (e.g., as in Variational Dropout).
     *
     * @param x The input tensor.
     * @return torch::Tensor The output tensor with adaptive dropout applied.
     */
    torch::Tensor adaptive_dropout(torch::Tensor x);

    /**
     * @brief A module wrapper for Adaptive Dropout with learnable dropout rates.
     *
     * Inherits from `xt::Module` and manages a learnable log-alpha parameter tensor (`log_alpha_`)
     * to adaptively tune dropout rates per feature during training.
     */
    struct AdaptiveDropout : xt::Module
    {
    public:
        /**
         * @brief Constructs an AdaptiveDropout module.
         *
         * @param probability_shape Shape of the learnable dropout parameter tensor (defaults to empty `{}`).
         * @param initial_dropout_rate Initial dropout probability value (defaults to 0.05).
         */
        explicit AdaptiveDropout(c10::IntArrayRef probability_shape = {}, double initial_dropout_rate = 0.05);

        /**
         * @brief Performs the forward pass for the AdaptiveDropout module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The resulting output tensor wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    private:
        /**
         * @brief Learnable parameter tensor representing log-alpha (log-variance ratio) for adaptive dropout.
         */
        torch::Tensor log_alpha_;
    };
}