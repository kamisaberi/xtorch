#pragma once

#include "../common.h"

namespace xt::transforms::general { // This is a general data/vector transform

    /**
     * @class StyleMixing
     * @brief A transform that implements StyleGAN-style mixing regularization.
     *
     * This technique improves generator quality by creating new images from a mix
     * of two different style vectors. It takes two style vectors (`w1` and `w2`)
     * and a random crossover point, and creates a new set of layer-specific styles
     * that can be fed into a StyleGAN generator.
     *
     * This transform operates on style vectors, not images directly.
     */
    class StyleMixing : public xt::Module {
    public:
        /**
         * @brief Default constructor. Assumes a 14-layer generator and 90% probability.
         */
        StyleMixing();

        /**
         * @brief Constructs the StyleMixing transform.
         * @param p The probability of applying style mixing.
         * @param n_layers The total number of style inputs in the generator model
         *                 (e.g., 14 for StyleGAN2 at 256x256, 18 for 1024x1024).
         */
        StyleMixing(float p, int n_layers);

        /**
         * @brief Executes the style mixing operation.
         * @param tensors An initializer list containing two tensors:
         *                1. The primary style vector(s) (e.g., shape [B, 512] or [B, n_layers, 512])
         *                2. The secondary style vector(s) to mix in.
         * @return An std::any containing the resulting mixed style tensor of shape
         *         [B, n_layers, 512].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float p_;
        int n_layers_;
    };

} // namespace xt::transforms::general