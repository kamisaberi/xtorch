#pragma once
#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class VerticalFlip
     * @brief An image transformation that flips an image vertically.
     *
     * This data augmentation technique flips the image around the horizontal
     * x-axis (top becomes bottom and vice-versa). This implementation uses
     * OpenCV's flip function for the underlying computation.
     */
    class VerticalFlip : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        VerticalFlip();

        /**
         * @brief Executes the vertical flip operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting vertically-flipped
         *         torch::Tensor with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::image