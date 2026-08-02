#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Varifocal Loss (VFL) function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Varifocal Loss (VFL) on an input tensor.
     *
     * Varifocal Loss (Zhang et al., CVPR 2021) is an asymmetric, IoU-aware loss function designed
     * for dense object detection. It weights positive samples using continuous target IoU scores
     * and negative samples using Focal Loss scaling to balance training:
     * \f$ \text{VFL}(p, q) = \begin{cases} -q (q \log(p) + (1-q) \log(1-p)) & \text{if } q > 0 \\ -\alpha p^\gamma \log(1-p) & \text{if } q = 0 \end{cases} \f$
     *
     * @param x Input tensor containing predicted probabilities/logits and target IoU scores.
     * @return torch::Tensor The computed Varifocal loss tensor.
     */
    torch::Tensor varifocal_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Varifocal Loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class VarifocalLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the VarifocalLoss module.
         */
        VarifocalLoss() = default;

        /**
         * @brief Performs the forward pass for the VarifocalLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}