#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomGridDropout
     * @brief Applies a grid-like dropout to an image.
     *
     * This transform, also known as GridMask, drops out (sets to a fill value)
     * multiple square regions in a grid pattern. The grid itself is randomly
     * positioned. This is an effective augmentation for improving model robustness.
     */
    class RandomGridDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. A common setup with a grid cycle of 100 pixels,
         *        dropout ratio of 0.5, and 50% application probability.
         */
        RandomGridDropout();

        /**
         * @brief Constructs the RandomGridDropout transform.
         *
         * @param ratio The ratio of the grid hole size to the grid cycle size.
         *              Must be between 0.0 and 1.0. A ratio of 0.5 means the
         *              holes and the remaining parts of the grid are equal in size.
         * @param grid_size A pair `{min, max}` specifying the range for the random
         *                  grid cycle size (d) in pixels. The grid hole size will be
         *                  `d * ratio`.
         * @param fill The constant value used for the dropped out regions.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomGridDropout(
            double ratio,
            std::pair<int, int> grid_size,
            double fill = 0.0,
            double p = 0.5
        );

        /**
         * @brief Executes the random grid dropout.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double ratio_;
        std::pair<int, int> grid_size_;
        double fill_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image