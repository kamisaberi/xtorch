#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomBrightnessContrast
     * @brief Randomly changes the brightness and contrast of an image.
     *
     * This transform applies a random brightness and contrast adjustment to an image
     * using the formula: `output = image * contrast_factor + brightness_factor`.
     * The factors are chosen randomly from user-defined ranges.
     * The operation is only applied with a given probability `p`.
     */
    class RandomBrightnessContrast : public xt::Module {
    public:
        /**
         * @brief Default constructor. Allows for a brightness change of +/- 20%
         *        and a contrast change of +/- 20%, applied with 50% probability.
         */
        RandomBrightnessContrast();

        /**
         * @brief Constructs the RandomBrightnessContrast transform.
         *
         * @param brightness_limit A non-negative float. The brightness factor will be
         *        randomly chosen from `[-limit, limit]`.
         * @param contrast_limit A non-negative float. The contrast factor will be
         *        randomly chosen from `[1.0 - limit, 1.0 + limit]`.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomBrightnessContrast(
            double brightness_limit,
            double contrast_limit,
            double p = 0.5
        );

        /**
         * @brief Executes the random brightness and contrast adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting torch::Tensor with the same
         *         shape and type as the input. The image may be unchanged if the
         *         random probability check fails.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double brightness_limit_;
        double contrast_limit_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image