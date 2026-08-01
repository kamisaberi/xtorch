#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss on an input tensor.
     *
     * NT-Xent Loss (Chen et al., SimCLR, ICML 2020) is a contrastive loss function evaluated on
     * L2-normalized feature representations using cosine similarity scaled by a temperature hyperparameter \f$\tau\f$:
     * \f$ \ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{I}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)} \f$
     *
     * @param x Input tensor containing feature representations or pairwise similarity matrices.
     * @return torch::Tensor The computed NT-Xent loss tensor.
     */
    torch::Tensor nt_xent_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the NT-Xent loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class NTXentLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the NTXentLoss module.
         */
        NTXentLoss() = default;

        /**
         * @brief Performs the forward pass for the NTXentLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}