#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Dynamic Smooth L1 loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Dynamic Smooth L1 Loss on an input error/residual tensor.
     *
     * Dynamic Smooth L1 Loss dynamically adapts the threshold hyperparameter (\f$\beta\f$)
     * separating L1 and L2 loss regimes according to training error statistics (e.g., as in Dynamic R-CNN),
     * providing optimal regression gradients across training stages.
     *
     * @param x The input error or residual tensor.
     * @return torch::Tensor The computed Dynamic Smooth L1 loss tensor.
     */
    torch::Tensor dynamic_smooth_l1_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Dynamic Smooth L1 loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DynamicSmoothL1Loss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the DynamicSmoothL1Loss module.
         */
        DynamicSmoothL1Loss() = default;

        /**
         * @brief Performs the forward pass for the DynamicSmoothL1Loss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}