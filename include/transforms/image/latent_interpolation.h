#pragma once

#include "../common.h"


namespace xt::transforms::general { // This is a general data transform, not just for images

    /**
     * @class LatentInterpolation
     * @brief A data augmentation transform that interpolates between samples in the latent space.
     *
     * This technique mixes data by performing a linear interpolation between the
     * latent feature vectors of two different samples from a batch. The corresponding
     * labels are also mixed with the same interpolation factor.
     *
     * This transform must be applied to an entire batch of data, including the
     * latent vectors and their corresponding labels.
     */
    class LatentInterpolation : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses alpha=1.0 and applies transform 50% of the time.
         */
        LatentInterpolation();

        /**
         * @brief Constructs the LatentInterpolation transform.
         * @param alpha The alpha parameter for the Beta(alpha, alpha) distribution,
         *              which determines the interpolation factor (lambda). A value
         *              of 1.0 gives a uniform distribution.
         * @param p The probability of applying the transform to a given batch.
         */
        LatentInterpolation(float alpha, float p = 0.5f);

        /**
         * @brief Executes the latent space interpolation.
         * @param tensors An initializer list containing exactly two tensors:
         *                1. A batch of latent vectors (2D, [B, LatentDim])
         *                2. A batch of corresponding labels (1D for int labels, or 2D for one-hot)
         * @return An std::any containing a std::pair<torch::Tensor, torch::Tensor>:
         *         {mixed_latents_batch, mixed_labels_batch}.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float alpha_;
        float p_;
    };

} // namespace xt::transforms::general