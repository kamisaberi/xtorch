#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the UPIT (Unsupervised Permutation Invariant Training) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the UPIT (Unsupervised Permutation Invariant Training) loss on an input tensor.
     *
     * UPIT loss evaluates a permutation-invariant or unsupervised translation objective,
     * handling label ambiguity across multiple permutation channels, sources, or domains.
     *
     * @param x Input tensor containing predictions, residuals, or logit scores.
     * @return torch::Tensor The computed UPIT loss tensor.
     */
    torch::Tensor upit_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the UPIT loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class UPITLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the UPITLoss module.
         */
        UPITLoss() = default;

        /**
         * @brief Performs the forward pass for the UPITLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}