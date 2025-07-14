#pragma once
#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Rotation
     * @brief Rotates an image by a fixed angle.
     *
     * This transform rotates the image by a specified angle. This is a
     * deterministic operation. The rotation is performed around the
     * center of the image.
     */
    class Rotation : public xt::Module {
    public:
        /**
         * @brief Default constructor. Does nothing as an angle is required.
         */
        Rotation();

        /**
         * @brief Constructs the Rotation transform.
         *
         * @param degrees The angle of rotation in degrees. Positive values mean
         *                counter-clockwise rotation.
         * @param expand If true, expands the output image to make it large enough
         *               to contain the entire rotated image. If false, keeps the
         *               original image size and crops the rotated image.
         * @param fill A vector representing the color to fill new areas with.
         * @param interpolation The interpolation method to use.
         */
        explicit Rotation(
            double degrees,
            bool expand = false,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the rotation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting rotated torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double degrees_;
        bool expand_;
        cv::Scalar fill_color_;
        int interpolation_flag_;
    };

} // namespace xt::transforms::image