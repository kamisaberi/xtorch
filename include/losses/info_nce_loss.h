#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the InfoNCE (Information Noise-Contrastive Estimation) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the InfoNCE (Information Noise-Contrastive Estimation) loss on an input tensor.
     *
     * InfoNCE loss (Oord et al., 2018) is a foundational contrastive learning objective widely used
     * in self-supervised representation learning (e.g., CPC, SimCLR, MoCo, CLIP). It maximizes mutual
     * information between positive sample pairs relative to negative distractor samples:
     * \f$ L_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(q, k_+) / \tau)}{\sum_{i} \exp(\text{sim}(q, k_i) / \tau)} \f$
     *
     * @param x Input tensor containing similarity matrices or query-key matching logits.
     * @return torch::Tensor The computed InfoNCE loss tensor.
     */
    torch::Tensor info_nce_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the InfoNCE loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class InfoNCELoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the InfoNCELoss module.
         */
        InfoNCELoss() = default;

        /**
         * @brief Performs the forward pass for the InfoNCELoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}