#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the WGAN-GP (Wasserstein GAN with Gradient Penalty) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the WGAN-GP (Wasserstein GAN with Gradient Penalty) loss on an input tensor.
     *
     * WGAN-GP Loss (Gulrajani et al., NeurIPS 2017) enforces a 1-Lipschitz continuity constraint
     * on the critic by adding a gradient norm penalty evaluated on interpolated samples,
     * significantly stabilizing adversarial training dynamics:
     * \f$ L = \mathbb{E}_{\tilde{x} \sim P_g}[D(\tilde{x})] - \mathbb{E}_{x \sim P_r}[D(x)] + \lambda \mathbb{E}_{\hat{x} \sim P_{\hat{x}}}[(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2] \f$
     *
     * @param x Input tensor containing critic predictions for real and fake samples, and/or gradient norm penalties.
     * @return torch::Tensor The computed WGAN-GP loss tensor.
     */
    torch::Tensor wgan_gp_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the WGAN-GP loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class WGANGPLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the WGANGPLoss module.
         */
        WGANGPLoss() = default;

        /**
         * @brief Performs the forward pass for the WGANGPLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}