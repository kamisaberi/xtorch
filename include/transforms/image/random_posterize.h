#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomPosterize
     * @brief Randomly reduces the number of bits for each color channel in an image.
     *
     * This effect quantizes the color space of an image, giving it a flatter,
     * "poster-like" appearance. The strength of the effect is controlled by
     * randomly selecting the number of bits to keep for each channel from a
     * user-defined range.
     * The operation is applied with a given probability `p`.
     */
    class RandomPosterize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Randomly selects bits between 4 and 8,
         *        with a 50% probability.
         */
        RandomPosterize();

        /**
         * @brief Constructs the RandomPosterize transform.
         * @param bits_range A pair `{min, max}` specifying the range for the number
         *                   of bits to keep. Values must be integers between 1 and 8.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomPosterize(
            std::pair<int, int> bits_range,
            double p = 0.5
        );

        /**
         * @brief Executes the random posterization operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting posterized torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<int, int> bits_range_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image