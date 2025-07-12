#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomRotation
     * @brief Rotates an image by a random angle.
     *
     * This transform rotates the image by an angle randomly selected from a
     * given range. The rotation is performed around the center of the image.
     */
    class RandomRotation : public xt::Module {
    public:
        /**
         * @brief Default constructor. Rotates by an angle in [-10, 10] degrees.
         */
        RandomRotation();

        /**
         * @brief Constructs the RandomRotation transform.
         *
         * @param degrees A pair `{min, max}` specifying the range in degrees for
         *                the random rotation angle.
         * @param expand If true, expands the output image to make it large enough
         *               to contain the entire rotated image. If false, keeps the
         *               original image size and crops the rotated image.
         * @param fill A vector representing the color to fill new areas with.
         *             Should be in the [0, 1] range for float tensors.
         * @param interpolation The interpolation method to use.
         *                      Supported: "bilinear" (default), "nearest".
         */
        explicit RandomRotation(
            std::pair<double, double> degrees,
            bool expand = false,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random rotation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting rotated torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<double, double> degrees_;
        bool expand_;
        cv::Scalar fill_color_;
        int interpolation_flag_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image