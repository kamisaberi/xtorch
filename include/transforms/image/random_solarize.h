#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomSolarize
     * @brief Inverts all pixel values above a threshold, with a given probability.
     *
     * This transform applies solarization to an image by inverting pixel values
     * that are above a specified `threshold`. The formula is:
     * `output = input if input < threshold else 1.0 - input`.
     * The operation is defined for float tensors in the [0, 1] range.
     * The operation is applied with a given probability `p`.
     */
    class RandomSolarize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies solarization with a threshold of 0.5
         *        and a 50% probability.
         */
        RandomSolarize();

        /**
         * @brief Constructs the RandomSolarize transform.
         *
         * @param threshold A value in [0, 1] above which pixels will be inverted.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomSolarize(
            double threshold,
            double p = 0.5
        );

        /**
         * @brief Executes the random solarization.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting solarized torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double threshold_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image