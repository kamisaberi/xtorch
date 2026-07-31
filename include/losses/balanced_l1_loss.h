#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Balanced L1 loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Balanced L1 Loss on an input error/residual tensor.
     *
     * Balanced L1 Loss increases gradient contribution for inliers (small errors)
     * to promote balanced training in object detection tasks (e.g., in Libra R-CNN).
     *
     * @param x The input error or residual tensor.
     * @return torch::Tensor The computed Balanced L1 loss tensor.
     */
    torch::Tensor balanced_l1_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Balanced L1 loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class BalancedL1Loss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the BalancedL1Loss module.
         */
        BalancedL1Loss() = default;

        /**
         * @brief Performs the forward pass for the BalancedL1Loss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
    };
}