#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the HAPPIER loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the HAPPIER loss on an input tensor.
     *
     * HAPPIER loss evaluates a human-aligned perceptual or preference loss function
     * designed for human-aligned generative modeling, image quality assessment, or
     * preference-alignment optimization tasks.
     *
     * @param x Input tensor containing predictions, residuals, or logit scores.
     * @return torch::Tensor The computed HAPPIER loss tensor.
     */
    torch::Tensor happier_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the HAPPIER loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class HAPPIERLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the HAPPIERLoss module.
         */
        HAPPIERLoss() = default;

        /**
         * @brief Performs the forward pass for the HAPPIERLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}