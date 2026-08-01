#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the GAN Hinge loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Hinge Loss for Generative Adversarial Networks (GANs) on an input tensor.
     *
     * GAN Hinge Loss (popularized in Spectral Normalization GAN, SAGAN, and BigGAN) uses a margin-based
     * hinge penalty to bound discriminator predictions and stabilize adversarial training dynamics:
     * \f$ L_D = -\mathbb{E}_{x \sim p_{\text{data}}}[\min(0, -1 + D(x))] - \mathbb{E}_{z \sim p_z}[\min(0, -1 - D(G(z)))] \f$
     *
     * @param x Input tensor containing discriminator logits/predictions for real and/or generated samples.
     * @return torch::Tensor The computed GAN Hinge loss tensor.
     */
    torch::Tensor gan_hinge_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the GAN Hinge loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class GANHingeLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the GANHingeLoss module.
         */
        GANHingeLoss() = default;

        /**
         * @brief Performs the forward pass for the GANHingeLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}