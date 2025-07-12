#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomRotate45
     * @brief Randomly rotates an image by a multiple of 45 degrees.
     *
     * This transform selects a random angle from {0, 45, 90, 135, 180, 225, 270, 315}
     * and rotates the image by that amount. This is a common augmentation in tasks
     * where rotational invariance at these specific angles is important.
     */
    class RandomRotate45 : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a random 45-degree increment rotation.
         */
        RandomRotate45();

        /**
         * @brief Constructs the RandomRotate45 transform.
         *
         * @param expand If true, expands the output image to make it large enough
         *               to contain the entire rotated image. If false, keeps the
         *               original image size and crops the rotated image.
         * @param fill A vector representing the color to fill new areas with.
         *             Should be in the [0, 1] range for float tensors.
         * @param interpolation The interpolation method to use.
         *                      Supported: "bilinear" (default), "nearest".
         */
        explicit RandomRotate45(
            bool expand = true,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random 45-degree rotation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting rotated torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        bool expand_;
        cv::Scalar fill_color_;
        int interpolation_flag_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image