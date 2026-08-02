#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Self-Adjusting Smooth L1 loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Self-Adjusting Smooth L1 Loss on an input error/residual tensor.
     *
     * Self-Adjusting Smooth L1 Loss dynamically tracks regression error statistics during
     * training to automatically tune its loss parameters, balancing gradient contributions
     * between inliers and outliers for bounding box localization.
     *
     * @param x Input error or residual tensor.
     * @return torch::Tensor The computed Self-Adjusting Smooth L1 loss tensor.
     */
    torch::Tensor self_adjusting_smooth_l1_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Self-Adjusting Smooth L1 loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class SelfAdjustingSmoothL1Loss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the SelfAdjustingSmoothL1Loss module.
         */
        SelfAdjustingSmoothL1Loss() = default;

        /**
         * @brief Performs the forward pass for the SelfAdjustingSmoothL1Loss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}