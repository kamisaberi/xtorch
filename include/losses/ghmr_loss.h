#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the GHM-R (Gradient Harmonized Mechanism for Regression) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the GHM-R (Gradient Harmonized Mechanism - Regression) loss on an input tensor.
     *
     * GHM-R Loss (Li et al., AAAI 2019) applies gradient density harmonized weighting to smooth L1
     * regression loss, balancing gradient contributions from easy regression targets and hard outliers
     * in bounding box localization tasks.
     *
     * @param x Input tensor containing regression error residuals or prediction-target pairs.
     * @return torch::Tensor The computed GHM-R loss tensor.
     */
    torch::Tensor ghmr_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the GHM-R loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class GHMRLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the GHMRLoss module.
         */
        GHMRLoss() = default;

        /**
         * @brief Performs the forward pass for the GHMRLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}