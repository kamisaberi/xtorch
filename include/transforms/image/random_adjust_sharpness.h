#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomAdjustSharpness
     * @brief An image transformation that adjusts the sharpness of an image by a random factor.
     *
     * This transform randomly sharpens or blurs an image. The adjustment is controlled by a
     * `sharpness_factor`.
     * - A factor of 0.0 gives a blurred image.
     * - A factor of 1.0 gives the original image.
     * - Factors > 1.0 enhance sharpness.
     * The transform randomly picks a sharpness adjustment level from the range
     * `[max(0, 1 - sharpness_factor), 1 + sharpness_factor]`.
     * The operation is only applied with a given probability `p`.
     */
    class RandomAdjustSharpness : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a sharpness adjustment between 0.0 and 2.0
         * with a 50% probability. This means it can range from fully blurred to
         * noticeably sharpened.
         */
        RandomAdjustSharpness();

        /**
         * @brief Constructs the RandomAdjustSharpness transform.
         * @param sharpness_factor A non-negative float. The transform will randomly
         *        pick a sharpness level from `[max(0, 1-factor), 1+factor]`.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomAdjustSharpness(double sharpness_factor, double p = 0.5);

        /**
         * @brief Executes the random sharpness adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with the same
         *         shape and type as the input. The image may be unchanged if the
         *         random probability check fails or the chosen factor is 1.0.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double sharpness_factor_;
        double p_;
        // A separate random engine for each instance is good practice for thread safety.
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image