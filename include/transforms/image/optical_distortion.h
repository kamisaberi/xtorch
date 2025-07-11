#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class OpticalDistortion
     * @brief An image transformation that simulates lens distortion effects like
     *        barrel or pincushion distortion.
     *
     * This is a strong geometric augmentation that makes models more robust to
     * images taken with different types of camera lenses. This implementation
     * uses OpenCV's camera model functions to apply the distortion.
     */
    class OpticalDistortion : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses reasonable default parameters for a moderate
         *        barrel distortion.
         */
        OpticalDistortion();

        /**
         * @brief Constructs the OpticalDistortion transform.
         * @param distort_limit A value controlling the intensity of radial distortion.
         *                      Positive values create barrel distortion (bulging out),
         *                      negative values create pincushion distortion (caving in).
         * @param shift_limit A value controlling the intensity of tangential distortion,
         *                    which creates a slight "decentering" effect.
         * @param interpolation The interpolation method to use. "linear" is default.
         * @param border_mode The pixel extrapolation method. "constant" is default.
         * @param fill_value The value to use for padded areas if border_mode is "constant".
         */
        OpticalDistortion(
            float distort_limit = 0.5f,
            float shift_limit = 0.5f,
            const std::string& interpolation = "linear",
            const std::string& border_mode = "constant",
            float fill_value = 0.0f
        );

        /**
         * @brief Executes the optical distortion operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting distorted torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float distort_limit_;
        float shift_limit_;
        int interpolation_flag_;
        int border_mode_flag_;
        float fill_value_;
    };

} // namespace xt::transforms::image