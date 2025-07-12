#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomThinPlateSpline
     * @brief Applies a random non-rigid warp using Thin Plate Splines (TPS).
     *
     * This transform creates a smooth, "bendy" distortion in the image. It works
     * by defining a grid of control points on the image, randomly moving them,
     * and then calculating a smooth warp that maps the original grid to the
     * new distorted grid.
     */
    class RandomThinPlateSpline : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a 4x4 grid and a moderate distortion scale.
         */
        RandomThinPlateSpline();

        /**
         * @brief Constructs the RandomThinPlateSpline transform.
         *
         * @param grid_size The number of control points along each dimension. A value
         *                  of 4 creates a 4x4 grid.
         * @param distortion_scale A float that controls the magnitude of the random
         *                         displacement of the control points, relative to the
         *                         grid cell size.
         * @param p The probability of applying the transform. Must be in [0, 1].
         * @param fill A vector representing the color to fill new areas with.
         * @param interpolation The interpolation method to use.
         */
        explicit RandomThinPlateSpline(
            int grid_size = 4,
            double distortion_scale = 0.2,
            double p = 0.5,
            const std::vector<double>& fill = {0.0, 0.0, 0.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random TPS warp.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting warped torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int grid_size_;
        double distortion_scale_;
        double p_;
        cv::Scalar fill_color_;
        int interpolation_flag_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image