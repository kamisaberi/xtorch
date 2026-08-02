#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the ZLPR (Zero-Label Pairwise Ranking) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the ZLPR (Zero-Label Pairwise Ranking) loss on an input tensor.
     *
     * ZLPR Loss (Su et al., 2022) is an efficient pairwise ranking loss for multi-label classification.
     * It uses a virtual zero-label threshold to separate positive and negative categories into an
     * uncoupled, smooth log-sum-exp optimization objective:
     * \f$ L_{\text{ZLPR}} = \log \left( 1 + \sum_{i \in \Omega_{\text{pos}}} e^{-s_i} \right) + \log \left( 1 + \sum_{j \in \Omega_{\text{neg}}} e^{s_j} \right) \f$
     *
     * @param x Input tensor containing predicted class logits or classification scores.
     * @return torch::Tensor The computed ZLPR loss tensor.
     */
    torch::Tensor zlpr_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the ZLPR loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class ZLPRLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ZLPRLoss module.
         */
        ZLPRLoss() = default;

        /**
         * @brief Performs the forward pass for the ZLPRLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}