#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Early Exiting loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Early Exiting Loss on an input tensor.
     *
     * Early Exiting Loss aggregates or weights losses computed across multiple intermediate exit
     * heads in multi-exit neural network architectures (such as BranchyNet, DeeBERT, or FastBERT),
     * allowing adaptive inference and early termination.
     *
     * @param x Input tensor containing predictions, logits, or loss values from intermediate exit heads.
     * @return torch::Tensor The computed Early Exiting loss tensor.
     */
    torch::Tensor early_exiting_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Early Exiting loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class EarlyExitingLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the EarlyExitingLoss module.
         */
        EarlyExitingLoss() = default;

        /**
         * @brief Performs the forward pass for the EarlyExitingLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}