#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomContrast
     * @brief An image transformation that adjusts the contrast of an image by a random factor.
     *
     * This transform randomly adjusts the image contrast by blending it with its
     * grayscale mean. The adjustment is controlled by a `contrast_factor`.
     * - A factor of 0.0 gives a solid gray image (zero contrast).
     * - A factor of 1.0 gives the original image.
     * - Factors > 1.0 increase contrast.
     * The transform randomly picks a contrast adjustment level from the range
     * `[max(0, 1 - contrast_limit), 1 + contrast_limit]`.
     * The operation is only applied with a given probability `p`.
     */
    class RandomContrast : public xt::Module {
    public:
        /**
         * @brief Default constructor. Allows for a contrast adjustment between 0.8 and 1.2
         *        with a 50% probability.
         */
        RandomContrast();

        /**
         * @brief Constructs the RandomContrast transform.
         * @param contrast_limit A non-negative float. The transform will randomly
         *        pick a contrast level from `[max(0, 1-limit), 1+limit]`.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomContrast(double contrast_limit, double p = 0.5);

        /**
         * @brief Executes the random contrast adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting torch::Tensor with the same
         *         shape and type as the input. The image may be unchanged if the
         *         random probability check fails or the chosen factor is 1.0.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double contrast_limit_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image