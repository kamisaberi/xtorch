#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Multi-Task / Multi-Loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Multi-Task Loss on an input tensor.
     *
     * MultiLoss aggregates or dynamically weights multiple loss components (e.g., using homoscedastic
     * uncertainty weighting across task objectives) into a single unified objective for multi-task learning models.
     *
     * @param x Input tensor containing individual task loss values, predictions, or task targets.
     * @return torch::Tensor The computed unified multi-task loss tensor.
     */
    torch::Tensor multi_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the MultiLoss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network multi-task loss pipelines.
     */
    class MultiLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the MultiLoss module.
         */
        MultiLoss() = default;

        /**
         * @brief Performs the forward pass for the MultiLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}