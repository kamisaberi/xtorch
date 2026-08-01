#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the HBM (Hierarchical/Hybrid Boundary Matching) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the HBM (Hierarchical/Hybrid Boundary Matching) loss on an input tensor.
     *
     * HBM loss evaluates boundary matching and structural alignment loss for image segmentation,
     * boundary estimation, and keypoint/feature matching tasks.
     *
     * @param x Input tensor containing predictions, logits, or boundary matching features.
     * @return torch::Tensor The computed HBM loss tensor.
     */
    torch::Tensor hbm_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the HBM loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class HBMLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the HBMLoss module.
         */
        HBMLoss() = default;

        /**
         * @brief Performs the forward pass for the HBMLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}