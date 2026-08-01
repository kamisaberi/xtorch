#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Metric Mixup loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Metric Mixup Loss on an input tensor.
     *
     * Metric Mixup Loss applies Mixup interpolation principles to deep metric learning,
     * synthesizing intermediate embeddings and pairwise distance targets to promote manifold
     * smoothness and improve distance metric generalization.
     *
     * @param x Input tensor containing interpolated feature embeddings or distance predictions and targets.
     * @return torch::Tensor The computed Metric Mixup loss tensor.
     */
    torch::Tensor metric_mixup_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Metric Mixup loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class MetricMixupLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the MetricMixupLoss module.
         */
        MetricMixupLoss() = default;

        /**
         * @brief Performs the forward pass for the MetricMixupLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}