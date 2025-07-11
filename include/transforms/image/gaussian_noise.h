#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class GaussianNoise
     * @brief An image transformation that adds Gaussian (Normal) noise to an image.
     *
     * This is a common data augmentation technique used to improve a model's
     * robustness to noise and variations in input data.
     */
    class GaussianNoise : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a mean of 0 and a standard deviation of 0.1.
         */
        GaussianNoise();

        /**
         * @brief Constructs the GaussianNoise transform with specific parameters.
         * @param mean The mean of the Gaussian distribution from which to draw noise.
         * @param sigma The standard deviation of the Gaussian distribution.
         */
        GaussianNoise(double mean, double sigma);

        /**
         * @brief Executes the noise addition operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting noisy torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double mean_;
        double sigma_;
    };

} // namespace xt::transforms::image