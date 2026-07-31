#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Cycle Consistency loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Cycle Consistency Loss on an input tensor.
     *
     * Cycle Consistency Loss enforces that translating a sample across domain transformations
     * and back recovers the original input (as popularized in CycleGAN architectures).
     *
     * @param x The input tensor representing reconstruction error or tensor difference.
     * @return torch::Tensor The computed Cycle Consistency loss tensor.
     */
    torch::Tensor cycle_consistency_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Cycle Consistency loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class CycleConsistencyLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the CycleConsistencyLoss module.
         */
        CycleConsistencyLoss() = default;

        /**
         * @brief Performs the forward pass for the CycleConsistencyLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}