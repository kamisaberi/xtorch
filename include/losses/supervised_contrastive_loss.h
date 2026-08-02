#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Supervised Contrastive (SupCon) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Supervised Contrastive (SupCon) Loss on an input tensor.
     *
     * Supervised Contrastive Loss (Khosla et al., NeurIPS 2020) extends self-supervised contrastive
     * learning (e.g., InfoNCE/SimCLR) to supervised settings. It pulls together embeddings of all
     * samples sharing the same class label while pushing away embeddings from different classes.
     *
     * @param x Input tensor containing normalized feature embeddings or pairwise similarity scores.
     * @return torch::Tensor The computed Supervised Contrastive loss tensor.
     */
    torch::Tensor supervised_contrastive_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Supervised Contrastive loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class SupervisedContrastiveLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the SupervisedContrastiveLoss module.
         */
        SupervisedContrastiveLoss() = default;

        /**
         * @brief Performs the forward pass for the SupervisedContrastiveLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}