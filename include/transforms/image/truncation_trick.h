#pragma once

#include "../common.h"


namespace xt::transforms::general { // This is a general data/vector transform

    /**
     * @class TruncationTrick
     * @brief A transform that applies the truncation trick to GAN latent vectors.
     *
     * This technique is used during GAN inference to improve the quality of generated
     * samples by truncating the latent space. It interpolates a given latent vector
     * towards the average latent vector, reducing the impact of outlier latents that
     * might produce low-quality images.
     *
     * This transform operates on latent vectors, not images directly.
     */
    class TruncationTrick : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized (and unusable) transform.
         */
        TruncationTrick();

        /**
         * @brief Constructs the TruncationTrick transform.
         * @param w_avg The average latent vector, pre-computed from the training data.
         *              It should have the shape [latent_dim] or [1, latent_dim].
         * @param truncation_psi A value between 0.0 and 1.0 controlling the amount
         *                       of truncation. 0.0 means all outputs are the average,
         *                       1.0 means no truncation.
         */
        TruncationTrick(torch::Tensor w_avg, double truncation_psi = 0.7);

        /**
         * @brief Executes the truncation operation.
         * @param tensors An initializer list expected to contain a single batch of
         *                latent vectors of shape [B, LatentDim].
         * @return An std::any containing the resulting truncated latent vectors.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor w_avg_;
        double truncation_psi_;
    };

} // namespace xt::transforms::general