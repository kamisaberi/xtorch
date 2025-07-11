#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class Upscale
     * @brief An image transformation that upscales an image by a given factor.
     *
     * This transform increases the height and width of an image while maintaining its
     * aspect ratio. It's useful for increasing resolution before further processing
     * or for tasks like super-resolution. This implementation uses OpenCV's resize
     * function with high-quality interpolation methods.
     */
    class Upscale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a scale factor of 2.0 (doubles the size).
         */
        Upscale();

        /**
         * @brief Constructs the Upscale transform with a specific scale factor.
         * @param scale_factor A factor by which to scale the image dimensions.
         *                     Must be greater than or equal to 1.0.
         * @param interpolation The interpolation method to use. "cubic" or "linear"
         *                      are recommended for upscaling.
         */
        Upscale(double scale_factor, const std::string& interpolation = "cubic");

        /**
         * @brief Executes the upscaling operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting upscaled torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double scale_factor_;
        int interpolation_flag_; // OpenCV interpolation flag
    };

} // namespace xt::transforms::image