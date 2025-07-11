#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class GlassBlur
     * @brief An image transformation that simulates viewing an image through glass.
     *
     * This is a strong corruption/augmentation that works by shuffling pixels
     * locally and then applying a Gaussian blur. It is used to test model
     * robustness against complex, non-linear distortions.
     * This implementation uses OpenCV for the underlying computation.
     */
    class GlassBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters for a moderate effect.
         */
        GlassBlur();

        /**
         * @brief Constructs the GlassBlur transform with specific parameters.
         * @param sigma The standard deviation of the Gaussian kernel for the final blur.
         * @param max_delta The maximum distance (in pixels) for shuffling pixels.
         * @param iterations The number of times to apply the pixel shuffling process.
         */
        GlassBlur(double sigma, int max_delta, int iterations);

        /**
         * @brief Executes the glass blur operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double sigma_;
        int max_delta_;
        int iterations_;
    };

} // namespace xt::transforms::image