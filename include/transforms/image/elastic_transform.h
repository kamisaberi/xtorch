#pragma once

#include "../common.h"


namespace xt::transforms::image
{
    /**
     * @class ElasticTransform
     * @brief An image transformation that applies elastic distortions.
     *
     * This is a powerful data augmentation technique that simulates non-rigid
     * deformations. It works by creating a smooth, random displacement field
     * and applying it to the image. This implementation uses OpenCV for the
     * underlying computation.
     */
    class ElasticTransform : public xt::Module
    {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters.
         */
        ElasticTransform();

        /**
         * @brief Constructs the ElasticTransform with specific parameters.
         * @param alpha The scaling factor for the displacement field, controlling
         *              the intensity of the distortion.
         * @param sigma The standard deviation of the Gaussian kernel used to smooth
         *              the displacement field. Controls the "elasticity".
         * @param interpolation The interpolation method to use. "linear" is default.
         * @param border_mode The pixel extrapolation method. "constant" is default.
         * @param fill_value The value to use for padded areas if border_mode is "constant".
         */
        ElasticTransform(
            double alpha,
            double sigma,
            const std::string& interpolation = "linear",
            const std::string& border_mode = "constant",
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the elastic distortion operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with the
         *         same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double alpha_;
        double sigma_;
        int interpolation_flag_;
        int border_mode_flag_;
        float fill_value_;
    };
} // namespace xt::transforms::image
