#pragma once

#include "common.h"


/**
 * @file
 * @brief Header file for the Adaptive Loss function and module.
 */

namespace xt::losses {
    /**
     * @brief Computes the General / Adaptive Robust Loss on an input residual tensor.
     *
     * The adaptive loss function generalizes a broad family of robust loss functions
     * (e.g., L2, L1, Charbonnier, Cauchy, Geman-McClure, Welsch) into a single form,
     * allowing the robustness characteristics of the loss to adapt dynamically.
     *
     * @param x The input error/residual tensor.
     * @return torch::Tensor The computed loss tensor.
     */
    torch::Tensor adaptive_loss(torch::Tensor x);


    /**
     * @brief A module wrapper for the Adaptive Loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class AdaptiveLoss : xt::Module {
    public:
        /**
         * @brief Default constructor for the AdaptiveLoss module.
         */
        AdaptiveLoss() = default;

        /**
         * @brief Performs the forward pass for the AdaptiveLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;
    private:
    };


}