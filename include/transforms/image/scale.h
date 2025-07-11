#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class Scale
     * @brief An image transformation that resizes an image by a given scale factor.
     *
     * This transform uniformly scales the height and width of an image while
     * maintaining its aspect ratio. It can be used for both upscaling (> 1.0)
     * and downscaling (< 1.0). This implementation uses OpenCV's resize function.
     */
    class Scale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Does nothing (identity transform, scale=1.0).
         */
        Scale();

        /**
         * @brief Constructs the Scale transform with a specific scale factor.
         * @param scale_factor A factor by which to scale the image dimensions.
         * @param interpolation The interpolation method to use. "linear" is a good
         *                      default. For downscaling, "area" is best. For upscaling,
         *                      "cubic" or "linear" are good.
         */
        explicit Scale(double scale_factor, const std::string& interpolation = "linear");

        /**
         * @brief Executes the scaling operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting scaled torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double scale_factor_;
        int interpolation_flag_; // OpenCV interpolation flag
    };

} // namespace xt::transforms::image