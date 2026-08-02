#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the PIoU (Pixel / Penalized Intersection over Union) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the PIoU (Pixel / Penalized Intersection over Union) loss on an input tensor.
     *
     * PIoU Loss (Chen et al., ECCV 2020) evaluates an IoU-based loss formulation optimized for oriented
     * object detection and pixel-level overlap, accounting for bounding box orientation angles and
     * high aspect ratio targets.
     *
     * @param x Input tensor containing predicted and target bounding boxes or pixel overlap predictions.
     * @return torch::Tensor The computed PIoU loss tensor.
     */
    torch::Tensor piou_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the PIoU loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class PIoULoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the PIoULoss module.
         */
        PIoULoss() = default;

        /**
         * @brief Performs the forward pass for the PIoULoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}