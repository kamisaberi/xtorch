#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Proxy-Anchor loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Proxy-Anchor Loss on an input tensor.
     *
     * Proxy-Anchor Loss (Kim et al., CVPR 2020) treats each proxy anchor as a query,
     * pulling positive feature samples while pushing away negative samples based on relative
     * sample difficulty in deep metric learning.
     *
     * @param x Input tensor containing feature embeddings or similarity scores with proxy anchors.
     * @return torch::Tensor The computed Proxy-Anchor loss tensor.
     */
    torch::Tensor proxy_anchor_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Proxy-Anchor loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class ProxyAnchorLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ProxyAnchorLoss module.
         */
        ProxyAnchorLoss() = default;

        /**
         * @brief Performs the forward pass for the ProxyAnchorLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}