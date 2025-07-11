#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class GridDistortion
     * @brief An image transformation that applies grid-based non-linear distortions.
     *
     * This augmentation technique overlays a grid on the image, randomly displaces
     * the grid points, and then smoothly remaps the image to this distorted grid.
     * It creates complex "wavy" distortions. This implementation uses OpenCV.
     */
    class GridDistortion : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters for a moderate effect.
         */
        GridDistortion();

        /**
         * @brief Constructs the GridDistortion transform.
         * @param num_steps The number of grid cells on each side.
         * @param distort_limit A factor controlling the maximum random displacement of
         *                      grid points, as a fraction of the grid cell size.
         * @param interpolation The interpolation method to use. "linear" is default.
         * @param border_mode The pixel extrapolation method. "constant" is default.
         * @param fill_value The value to use for padded areas if border_mode is "constant".
         */
        GridDistortion(
            int num_steps,
            float distort_limit,
            const std::string& interpolation = "linear",
            const std::string& border_mode = "constant",
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the grid distortion operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int num_steps_;
        float distort_limit_;
        int interpolation_flag_;
        int border_mode_flag_;
        float fill_value_;
    };

} // namespace xt::transforms::image