#pragma once
#include "../common.h"

#pragma once

#include <random>
#include <utility>
#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomGaussianBlur
     * @brief Applies a Gaussian blur to an image with a random sigma.
     *
     * This transform blurs an image using a Gaussian filter with a specified
     * kernel size and a standard deviation (sigma) chosen randomly from a
     * given range.
     * The operation is applied with a given probability `p`.
     */
    class RandomGaussianBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a 7x7 kernel and a sigma range of [0.1, 2.0]
         *        with a 50% probability.
         */
        RandomGaussianBlur();

        /**
         * @brief Constructs the RandomGaussianBlur transform.
         *
         * @param kernel_size The size of the Gaussian kernel. Must be a positive, odd integer.
         * @param sigma_range A pair `{min, max}` specifying the range for the random
         *                    sigma value. A larger sigma results in a stronger blur.
         *                    Values must be non-negative.
         * @param p The probability of the transform being applied. Must be in [0, 1].
         */
        explicit RandomGaussianBlur(
            int kernel_size,
            std::pair<double, double> sigma_range,
            double p = 0.5
        );

        /**
         * @brief Executes the random Gaussian blur.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int kernel_size_;
        std::pair<double, double> sigma_range_;
        double p_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image

