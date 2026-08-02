#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Triplet Loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Triplet Margin Loss on an input tensor.
     *
     * Triplet Loss (Schroff et al., FaceNet, CVPR 2015) minimizes the distance between an anchor
     * and a positive sample while enforcing a margin separation from a negative sample:
     * \f$ L(A, P, N) = \max(d(a, p) - d(a, n) + \text{margin}, 0) \f$
     *
     * @param x Input tensor containing anchor, positive, and negative embeddings or pairwise distances.
     * @return torch::Tensor The computed Triplet loss tensor.
     */
    torch::Tensor triplet_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Triplet Loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class TripletLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the TripletLoss module.
         */
        TripletLoss() = default;

        /**
         * @brief Performs the forward pass for the TripletLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}