#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the FLIP perceptually-based image loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the FLIP loss on an input image/difference tensor.
     *
     * FLIP (Andersson et al., NVIDIA) is a perceptual image difference metric designed for rendered
     * and synthesized images, modeling human visual system (HVS) responses such as spatial frequency
     * filtering and color perception.
     *
     * @param x Input tensor representing difference maps, predicted images, or target pairs.
     * @return torch::Tensor The computed FLIP loss tensor.
     */
    torch::Tensor flip_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the FLIP loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class FLIPLoss : xt::Module
    {
    public:

        /**
         * @brief Default constructor for the FLIPLoss module.
         */
        FLIPLoss() = default;

        /**
         * @brief Performs the forward pass for the FLIPLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}