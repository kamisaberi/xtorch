#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the ElasticFace (Elastic Margin Loss) function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the ElasticFace loss on an input tensor.
     *
     * ElasticFace (Boutros et al., CVPR 2022) introduces elastic margins by sampling angular/cosine
     * margins from a normal distribution around a mean value, relaxing fixed-margin constraints and
     * improving feature representations in deep face recognition and metric learning.
     *
     * @param x Input tensor containing feature embeddings or logits.
     * @return torch::Tensor The computed ElasticFace loss tensor.
     */
    torch::Tensor elastic_face(torch::Tensor x);

    /**
     * @brief A module wrapper for the ElasticFace loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class ElasticFace : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the ElasticFace module.
         */
        ElasticFace() = default;

        /**
         * @brief Performs the forward pass for the ElasticFace module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}