#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Perspective
     * @brief An image transformation that applies a random four-point perspective transform.
     *
     * This transform simulates viewing an image from different camera angles by
     * randomly moving the corners of the image and remapping the pixels.
     * It is a strong geometric augmentation. This implementation uses OpenCV.
     */
    class Perspective : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a moderate distortion scale.
         */
        Perspective();

        /**
         * @brief Constructs the Perspective transform.
         * @param distortion_scale A factor controlling the maximum random displacement
         *                         of the corners, as a fraction of image size.
         *                         A value of 0.5 means corners can move up to 50%
         *                         of the image's height/width.
         * @param p The probability of applying the transform.
         * @param interpolation The interpolation method to use.
         * @param fill_value The value to use for padded areas.
         */
        Perspective(
            float distortion_scale = 0.5f,
            float p = 0.5f,
            const std::string& interpolation = "linear",
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the perspective distortion operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float distortion_scale_;
        float p_;
        int interpolation_flag_;
        float fill_value_;
    };

} // namespace xt::transforms::image