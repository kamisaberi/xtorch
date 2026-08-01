#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Dice-BCE (Dice + Binary Cross Entropy) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the combined Dice and Binary Cross-Entropy (Dice-BCE) loss on an input tensor.
     *
     * Dice-BCE loss combines Dice Loss (which maximizes spatial overlap) with Binary Cross-Entropy
     * Loss (which optimizes per-pixel classification accuracy), commonly used in image segmentation tasks.
     *
     * @param x Input tensor containing predictions and targets or logits.
     * @return torch::Tensor The computed Dice-BCE loss tensor.
     */
    torch::Tensor dice_bce_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Dice-BCE loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DiceBCELoss : xt::Module
    {
    public:

        /**
         * @brief Default constructor for the DiceBCELoss module.
         */
        DiceBCELoss() = default;

        /**
         * @brief Performs the forward pass for the DiceBCELoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}