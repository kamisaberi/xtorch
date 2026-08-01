#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the GHM-C (Gradient Harmonized Mechanism for Classification) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the GHM-C (Gradient Harmonized Mechanism - Classification) loss on an input tensor.
     *
     * GHM-C Loss (Li et al., AAAI 2019) dynamically weights samples based on gradient density,
     * harmonizing gradient contributions by down-weighting both huge quantities of easy examples
     * and extreme outliers.
     *
     * @param x Input tensor containing predicted logits/probabilities and ground truth labels.
     * @return torch::Tensor The computed GHM-C loss tensor.
     */
    torch::Tensor ghmc_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the GHM-C loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class GHMCLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the GHMCLoss module.
         */
        GHMCLoss() = default;

        /**
         * @brief Performs the forward pass for the GHMCLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}