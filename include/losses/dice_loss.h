#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Dice loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Sørensen–Dice Loss on an input tensor.
     *
     * Dice Loss measures spatial overlap between predicted probability maps and ground truth masks:
     * \f$ L_{\text{Dice}} = 1 - \frac{2 |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|} \f$
     * It is widely used in image segmentation tasks to effectively handle class imbalance.
     *
     * @param x Input tensor containing predictions and targets or logits.
     * @return torch::Tensor The computed Dice loss tensor.
     */
    torch::Tensor dice_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Dice loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DiceLoss : xt::Module
    {
    public:

        /**
         * @brief Default constructor for the DiceLoss module.
         */
        DiceLoss() = default;

        /**
         * @brief Performs the forward pass for the DiceLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}