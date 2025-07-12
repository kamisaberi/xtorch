#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomGamma
     * @brief Applies a random gamma correction to an image.
     *
     * This transform adjusts the brightness of an image non-linearly using the
     * formula `output = input ^ gamma`. The `gamma` value is chosen randomly
     * from a specified range.
     * - A gamma of 1.0 results in no change.
     * - A gamma < 1.0 makes the image brighter.
     * - A gamma > 1.0 makes the image darker.
     * The operation is applied with a given probability `p`.
     */
    class RandomGamma : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a gamma range of [0.8, 1.2] and a 50% probability.
         */
        RandomGamma();

        /**
         * @brief Constructs the RandomGamma transform.
         *
         * @param gamma_range A pair `{min, max}` specifying the range for the random
         *                    gamma value. Values must be positive.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomGamma(
            std::pair<double, double> gamma_range,
            double p = 0.5
        );

        /**
         * @brief Executes the random gamma correction.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) with float values in the range [0, 1].
         * @return An std::any containing the resulting gamma-corrected torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<double, double> gamma_range_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image