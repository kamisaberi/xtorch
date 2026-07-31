#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the ArcFace (Additive Angular Margin) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the ArcFace (Additive Angular Margin) loss on an input tensor.
     *
     * ArcFace introduces an additive angular margin penalty to the target angle between
     * embeddings and class weight vectors, maximizing target logit margin and enhancing
     * intra-class compactness and inter-class variance.
     *
     * @param x The input tensor (e.g., predicted logits or normalized features).
     * @return torch::Tensor The computed ArcFace loss tensor.
     */
    torch::Tensor arcface_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the ArcFace loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class ArcFaceLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ArcFaceLoss module.
         */
        ArcFaceLoss() = default;

        /**
         * @brief Performs the forward pass for the ArcFaceLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed ArcFace loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
    };
}