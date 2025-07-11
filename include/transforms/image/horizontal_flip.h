#pragma once
#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class HorizontalFlip
     * @brief An image transformation that flips an image horizontally.
     *
     * This is one of the most common and effective data augmentation techniques.
     * It flips the image around the vertical y-axis. This implementation
     * uses OpenCV's flip function for the underlying computation.
     */
    class HorizontalFlip : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        HorizontalFlip();

        /**
         * @brief Executes the horizontal flip operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting horizontally-flipped
         *         torch::Tensor with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::image