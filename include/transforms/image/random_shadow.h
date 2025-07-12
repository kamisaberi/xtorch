#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomShadow
     * @brief Adds a random shadow polygon to an image.
     *
     * This transform simulates a shadow being cast across the image. It does this
     * by generating a random quadrilateral, creating a mask from it, and
     * darkening the image within that mask.
     * The operation is applied with a given probability `p`.
     */
    class RandomShadow : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a shadow with an intensity between
         *        0.5 and 1.0, with a 50% probability.
         */
        RandomShadow();

        /**
         * @brief Constructs the RandomShadow transform.
         *
         * @param shadow_range A pair `{min, max}` specifying the range for the
         *                     shadow's intensity (how dark it is). A value of 1.0
         *                     is no shadow, while 0.0 is a black shadow.
         *                     Values must be in [0, 1].
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomShadow(
            std::pair<double, double> shadow_range,
            double p = 0.5
        );

        /**
         * @brief Executes the random shadow augmentation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<double, double> shadow_range_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image