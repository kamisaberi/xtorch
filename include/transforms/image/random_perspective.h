#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomPerspective
     * @brief Performs a random perspective transformation on an image.
     *
     * This transform simulates viewing an image from a different angle by
     * randomly moving the four corners of the image and then warping the
     * image to fit the new quadrilateral.
     */
    class RandomPerspective : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a moderate perspective distortion
         *        with 50% probability.
         */
        RandomPerspective();

        /**
         * @brief Constructs the RandomPerspective transform.
         *
         * @param distortion_scale A float in [0, 1] that controls the magnitude
         *                         of the perspective distortion. A value of 0.5
         *                         means the corners can move by up to 50% of the
         *                         image half-width/height.
         * @param p The probability of applying the transform. Must be in [0, 1].
         * @param fill A vector representing the color to fill new areas with.
         *             Should be in the [0, 1] range for float tensors.
         * @param interpolation The interpolation method to use.
         *                      Supported: "bilinear" (default), "nearest".
         */
        explicit RandomPerspective(
            double distortion_scale,
            double p = 0.5,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random perspective transformation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double distortion_scale_;
        double p_;
        cv::Scalar fill_color_;
        int interpolation_flag_;

        std::mt19937 gen_;
    };

} // namespace xt::transforms::image