#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Generalized Focal Loss (GFL) function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Generalized Focal Loss (GFL) on an input tensor.
     *
     * Generalized Focal Loss (Li et al., NeurIPS 2020) extends standard Focal Loss from discrete
     * one-hot labels to continuous target scores and general probability distributions, unifying
     * localization quality estimation and classification in object detection models.
     *
     * @param x Input tensor containing predicted logits and continuous target scores/distributions.
     * @return torch::Tensor The computed Generalized Focal loss tensor.
     */
    torch::Tensor generalized_focal_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Generalized Focal Loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class GeneralizedFocalLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the GeneralizedFocalLoss module.
         */
        GeneralizedFocalLoss() = default;

        /**
         * @brief Performs the forward pass for the GeneralizedFocalLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}