#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomBrightness
     * @brief An image transformation that adjusts the brightness of an image by a random factor.
     *
     * This transform randomly adjusts the brightness of an image. The adjustment is
     * controlled by a `brightness_factor`.
     * - A factor of 0.0 gives a black image.
     * - A factor of 1.0 gives the original image.
     * - Factors > 1.0 increase brightness.
     * The transform randomly picks a brightness adjustment level from the range
     * `[max(0, 1 - brightness_factor), 1 + brightness_factor]`.
     * The operation is only applied with a given probability `p`.
     */
    class RandomBrightness : public xt::Module {
    public:
        /**
         * @brief Default constructor. Allows for a brightness adjustment between 0.5 and 1.5
         *        with a 50% probability.
         */
        RandomBrightness();

        /**
         * @brief Constructs the RandomBrightness transform.
         * @param brightness_factor A non-negative float. The transform will randomly
         *        pick a brightness level from `[max(0, 1-factor), 1+factor]`.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomBrightness(double brightness_factor, double p = 0.5);

        /**
         * @brief Executes the random brightness adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting torch::Tensor with the same
         *         shape and type as the input. The image may be unchanged if the
         *         random probability check fails or the chosen factor is 1.0.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double brightness_factor_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image