#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Downscale
     * @brief An image transformation that downscales an image by a given factor.
     *
     * This transform reduces the height and width of an image while maintaining its
     * aspect ratio. It's useful for creating lower-resolution inputs or for certain
     * data augmentation techniques. This implementation uses OpenCV's resize
     * function with area-based interpolation for high-quality results.
     */
    class Downscale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a scale factor of 0.5.
         */
        Downscale();

        /**
         * @brief Constructs the Downscale transform with a specific scale factor.
         * @param scale_factor A factor by which to scale the image dimensions.
         *                     Must be between 0.0 and 1.0.
         * @param interpolation The interpolation method to use. "area" is recommended
         *                      for downscaling. Other options: "linear", "cubic".
         */
        Downscale(double scale_factor, const std::string& interpolation = "area");

        /**
         * @brief Executes the downscaling operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting downscaled torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double scale_factor_;
        int interpolation_flag_; // OpenCV interpolation flag
    };

} // namespace xt::transforms::image