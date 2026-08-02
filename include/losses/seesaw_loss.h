#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Seesaw loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Seesaw Loss on an input tensor.
     *
     * Seesaw Loss (Wang et al., CVPR 2021) dynamically re-balances negative gradient contributions
     * between tail (infrequent) and head (frequent) classes in long-tailed object detection and
     * instance segmentation using mitigation and compensation factors.
     *
     * @param x Input tensor containing predicted logits and ground truth class labels.
     * @return torch::Tensor The computed Seesaw loss tensor.
     */
    torch::Tensor seesaw_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Seesaw loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class SeesawLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the SeesawLoss module.
         */
        SeesawLoss() = default;

        /**
         * @brief Performs the forward pass for the SeesawLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}