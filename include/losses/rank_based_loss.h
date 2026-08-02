#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Rank-Based loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Rank-Based Loss on an input tensor.
     *
     * Rank-Based Loss optimizes ranking-specific performance metrics (such as Average Precision,
     * Recall@K, or Mean Reciprocal Rank) by penalizing ranking misorderings directly over
     * prediction logits or similarity scores.
     *
     * @param x Input tensor containing predicted scores, logits, or candidate similarity values.
     * @return torch::Tensor The computed Rank-Based loss tensor.
     */
    torch::Tensor rank_based_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Rank-Based loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class RankBasedLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the RankBasedLoss module.
         */
        RankBasedLoss() = default;

        /**
         * @brief Performs the forward pass for the RankBasedLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}