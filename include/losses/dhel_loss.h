#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the DHEL loss function and module.
 */

namespace xt::losses
{

    /**
     * @brief Computes the DHEL (Dual Hinge Exponential Loss) on an input tensor.
     *
     * DHEL loss evaluates a hinge-based robust loss function designed to produce
     * compact and robust representation representations in deep learning models.
     *
     * @param x Input tensor (e.g., error, prediction, or logit tensor).
     * @return torch::Tensor The computed DHEL loss tensor.
     */
    torch::Tensor dhel_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the DHEL loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DHELLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the DHELLoss module.
         */
        DHELLoss() = default;

        /**
         * @brief Performs the forward pass for the DHELLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}