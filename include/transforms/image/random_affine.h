#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class RandomAffine
     * @brief Applies a random affine transformation to an image.
     *
     * The transformation is a combination of rotation, translation, scaling, and
     * shearing, applied with a given probability. All transformations are
     * performed with respect to the center of the image.
     */
    class RandomAffine : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a transform with 50% probability
         *        but with all transformation parameters set to zero (i.e., it does nothing).
         */
        RandomAffine();

        /**
         * @brief Constructs the RandomAffine transform with detailed parameters.
         *
         * @param degrees The maximum absolute range for rotation in degrees.
         *                For example, `10` means a random rotation in `[-10, 10]`.
         * @param translate Optional pair of max fractional translations.
         *                  For example, `{0.1, 0.1}` allows horizontal and vertical
         *                  translation by up to 10% of the image width and height.
         * @param scale Optional range `{min, max}` for isotropic scaling factor.
         *              For example, `{0.8, 1.2}`.
         * @param shear Optional pair of max absolute shear ranges in degrees for x and y axes.
         *              For example, `{10, 10}` allows shearing in `[-10, 10]` on x-axis
         *              and `[-10, 10]` on y-axis.
         * @param p The probability of applying the transform. Must be in [0, 1].
         * @param fill A vector representing the color to fill new areas with.
         *             Should be in the [0, 1] range for float tensors.
         *             E.g., `{0.0, 0.0, 0.0}` for black.
         * @param interpolation The interpolation method to use.
         *                      Supported: "bilinear" (default), "nearest", "bicubic".
         */
        explicit RandomAffine(
            double degrees,
            std::optional<std::pair<double, double>> translate = std::nullopt,
            std::optional<std::pair<double, double>> scale = std::nullopt,
            std::optional<std::pair<double, double>> shear = std::nullopt,
            double p = 0.5,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random affine transformation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor. The image may be
         *         unchanged if the probability check fails.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Generates and returns a 2x3 affine transformation matrix
        cv::Mat get_random_transform_matrix(int width, int height);

        double degrees_;
        std::optional<std::pair<double, double>> translate_;
        std::optional<std::pair<double, double>> scale_;
        std::optional<std::pair<double, double>> shear_;
        double p_;
        cv::Scalar fill_color_; // Store as cv::Scalar for direct use
        int interpolation_flag_; // Store as OpenCV flag for efficiency

        std::mt19937 gen_;
    };

} // namespace xt::transforms::image