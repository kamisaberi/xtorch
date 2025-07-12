#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomShortSideScale
     * @brief Scales an image by resizing its shorter side to a random value,
     *        maintaining the aspect ratio.
     *
     * This is a common augmentation for object detection models. It resizes the
     * image so that its shorter side is equal to a value chosen randomly from
     * a given range. The longer side is scaled proportionally. An optional
     * max size can be provided to cap the size of the longer side.
     */
    class RandomShortSideScale : public xt::Module {
    public:
        /**
         * @brief Default constructor. Scales the short side to a value between
         *        640 and 800 pixels.
         */
        RandomShortSideScale();

        /**
         * @brief Constructs the RandomShortSideScale transform.
         *
         * @param short_side_range A pair `{min, max}` specifying the range for the
         *                         target size of the shorter side.
         * @param max_size The maximum allowed size for the longer side. If the
         *                 proportional scaling results in a longer side greater
         *                 than this, the image will be downscaled to meet this
         *                 constraint. A value of 0 means no limit.
         * @param interpolation The interpolation method for resizing.
         *                      Supported: "bilinear" (default), "nearest", "bicubic".
         */
        explicit RandomShortSideScale(
            std::pair<int, int> short_side_range,
            int max_size = 0,
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
        std::pair<int, int> short_side_range_;
        int max_size_;
        std::string interpolation_mode_str_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image