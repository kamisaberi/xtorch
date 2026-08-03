#pragma once

#include "common.h"

/**
 * @file
 * @brief Header file for the AttentionDropout module.
 */

namespace xt::dropouts
{
    /**
     * @brief A module implementation for Attention Dropout.
     *
     * AttentionDropout randomly zero-masks elements of attention weights or probability matrices
     * during training with probability \p p_ to prevent co-adaptation in Transformer attention mechanisms.
     */
    struct AttentionDropout : xt::Module
    {
    public:
        /**
         * @brief Constructs an AttentionDropout module.
         *
         * @param p Probability of an element to be zeroed (defaults to 0.1).
         */
        AttentionDropout(double p = 0.1);

        /**
         * @brief Performs the forward pass for AttentionDropout.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The output attention tensor with dropout applied, wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Internal dropout probability factor.
         */
        double p_; // Probability of an element to be zeroed.
    };
}