#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Dual Softmax loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Dual Softmax Loss on an input score or similarity matrix.
     *
     * Dual Softmax applies softmax normalization independently along two dimensions (e.g., rows
     * and columns of a feature matching correlation matrix) to compute bidirectional matching
     * probability distributions (popularized in feature matching architectures such as LoFTR).
     *
     * @param x Input tensor containing similarity scores or matching logits.
     * @return torch::Tensor The computed Dual Softmax loss tensor.
     */
    torch::Tensor dual_softmax_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Dual Softmax loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DualSoftmaxLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the DualSoftmaxLoss module.
         */
        DualSoftmaxLoss() = default;

        /**
         * @brief Performs the forward pass for the DualSoftmaxLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}