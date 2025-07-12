#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomScale
     * @brief Resizes an image by a random scaling factor.
     *
     * This transform randomly selects a scaling factor from a given range and
     * resizes the image accordingly.
     */
    class RandomScale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Scales by a factor between 0.8 and 1.2.
         */
        RandomScale();

        /**
         * @brief Constructs the RandomScale transform.
         *
         * @param scale_range A pair `{min, max}` specifying the range for the random
         *                    scaling factor. Values must be positive.
         * @param interpolation The interpolation method for resizing.
         *                      Supported: "bilinear" (default), "nearest", "bicubic".
         */
        explicit RandomScale(
            std::pair<double, double> scale_range,
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random scaling.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting resized torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<double, double> scale_range_;
        std::string interpolation_mode_str_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image