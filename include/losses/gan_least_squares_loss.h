#pragma once
#include "common.h"


/**
 * @file
 * @brief Header file for the Least Squares GAN (LSGAN) loss function and module.
 */

namespace xt::losses
{
    /**
     * @brief Computes the Least Squares GAN (LSGAN) loss on an input tensor.
     *
     * LSGAN (Mao et al., ICCV 2017) replaces the standard binary cross-entropy loss in GANs
     * with a least-squares distance objective to prevent vanishing gradients and pull generated
     * samples toward the decision boundary:
     * \f$ L_D = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}}[(D(x) - b)^2] + \frac{1}{2} \mathbb{E}_{z \sim p_z}[(D(G(z)) - a)^2] \f$
     *
     * @param x Input tensor containing discriminator predictions/logits for real and/or fake samples.
     * @return torch::Tensor The computed LSGAN loss tensor.
     */
    torch::Tensor gan_least_squares_loss(torch::Tensor x);

    /**
     * @brief A module wrapper for the Least Squares GAN (LSGAN) loss function.
     *
     * Inherits from `xt::Module` to support dynamic execution within neural network loss pipelines.
     */
    class GANLeastSquaresLoss : xt::Module
    {
    public:
        /**
         * @brief Default constructor for the GANLeastSquaresLoss module.
         */
        GANLeastSquaresLoss() = default;

        /**
         * @brief Performs the forward pass for the GANLeastSquaresLoss module.
         *
         * @param tensors An initializer list of `std::any` containing arguments/tensors passed to the module.
         * @return std::any The computed loss wrapped in `std::any`.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any  override;


    private:
    };
}