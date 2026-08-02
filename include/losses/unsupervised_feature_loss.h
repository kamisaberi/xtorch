#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Unsupervised Feature loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Unsupervised Feature Loss on an input tensor.
     *
     * Unsupervised Feature Loss evaluates feature consistency, representation variance, or
     * feature-matching objectives without requiring explicit target class labels (common in
     * self-supervised learning, domain adaptation, and feature reconstruction).
     *
     * @param x Input tensor containing feature representations, feature differences, or activations.
     * @return torch::Tensor The computed Unsupervised Feature loss tensor.
     */
    torch::Tensor unsupervised_feature_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Unsupervised Feature loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class UnsupervisedFeatureLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the UnsupervisedFeatureLoss module.
         */
        UnsupervisedFeatureLoss() = default;

        /**
         * @brief Performs the forward pass for the UnsupervisedFeatureLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}