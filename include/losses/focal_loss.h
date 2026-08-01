#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Focal loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Focal Loss on an input tensor.
     *
     * Focal Loss (Lin et al., ICCV 2017) dynamically scales cross-entropy loss based on
     * prediction confidence, down-weighting easy examples to focus model training on hard
     * negative examples and mitigate class imbalance:
     * \f$ \text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \f$
     *
     * @param x Input tensor containing predictions/logits and target labels.
     * @return torch::Tensor The computed Focal loss tensor.
     */
    torch::Tensor focal_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Focal loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class FocalLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the FocalLoss module.
         */
        FocalLoss() = default;

        /**
         * @brief Performs the forward pass for the FocalLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}