#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Lovász-Softmax loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Lovász-Softmax Loss on an input tensor.
     *
     * Lovász-Softmax Loss (Berman et al., CVPR 2018) provides a direct, tractable surrogate for
     * optimizing the multi-class Intersection-over-Union (IoU / Jaccard index) metric in semantic
     * segmentation using the Lovász extension of submodular set functions.
     *
     * @param x Input tensor containing multi-class predictions/logits and ground truth labels.
     * @return torch::Tensor The computed Lovász-Softmax loss tensor.
     */
    torch::Tensor lovasz_softmax_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Lovász-Softmax loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class LovaszSoftmaxLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the LovaszSoftmaxLoss module.
         */
        LovaszSoftmaxLoss() = default;

        /**
         * @brief Performs the forward pass for the LovaszSoftmaxLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}