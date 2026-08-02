#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Triplet Entropy loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Triplet Entropy Loss on an input tensor.
     *
     * Triplet Entropy Loss combines triplet metric learning relationships (evaluating
     * anchor-positive versus anchor-negative distances) with an entropy / cross-entropy
     * formulation, providing soft-margin optimization for deep metric learning and Person Re-ID.
     *
     * @param x Input tensor containing feature embeddings, pairwise distances, or triplet logits.
     * @return torch::Tensor The computed Triplet Entropy loss tensor.
     */
    torch::Tensor triplet_entropy_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Triplet Entropy loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class TripletEntropyLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the TripletEntropyLoss module.
         */
        TripletEntropyLoss() = default;

        /**
         * @brief Performs the forward pass for the TripletEntropyLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}