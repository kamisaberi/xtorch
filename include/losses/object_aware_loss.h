#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Object-Aware loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Object-Aware Loss on an input tensor.
     *
     * Object-Aware Loss focuses loss evaluation specifically on salient object regions
     * (e.g., bounding box regions or object mask proposals), suppressing background
     * noise to improve localization, feature representations, and segmentation accuracy.
     *
     * @param x Input tensor containing predictions, feature maps, or logits.
     * @return torch::Tensor The computed Object-Aware loss tensor.
     */
    torch::Tensor object_aware_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Object-Aware loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class ObjectAwareLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ObjectAwareLoss module.
         */
        ObjectAwareLoss() = default;

        /**
         * @brief Performs the forward pass for the ObjectAwareLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}