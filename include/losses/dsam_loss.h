#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the DSAM (Dual-Supervised Attention Mechanism) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the DSAM (Dual-Supervised Attention Mechanism / Deep Supervision Attention Module) loss on an input tensor.
     *
     * DSAM loss evaluates attention-guided supervision loss, guiding the network to focus on
     * salient feature regions while enforcing structural and boundary consistency.
     *
     * @param x Input tensor containing predictions and targets or attention feature maps.
     * @return torch::Tensor The computed DSAM loss tensor.
     */
    torch::Tensor dsam_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the DSAM loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class DSAMLoss : xt::Module
    {
    public:

        /**
         * @brief Default constructor for the DSAMLoss module.
         */
        DSAMLoss() = default;

        /**
         * @brief Performs the forward pass for the DSAMLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;

    private:
    };
}