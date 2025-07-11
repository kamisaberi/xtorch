#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class CropAndPad
     * @brief An image transformation that extracts a rectangular region and pads
     *        it if the crop extends beyond the image boundaries.
     *
     * This transform attempts to crop a region defined by a top-left corner
     * and a target size. If the specified region is partially or completely
     * outside the image, the output is padded with a fill value to match the
     * target size.
     */
    class CropAndPad : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized transform.
         */
        CropAndPad();

        /**
         * @brief Constructs the CropAndPad transform.
         * @param top The vertical coordinate of the top-left corner of the crop.
         * @param left The horizontal coordinate of the top-left corner of the crop.
         * @param height The target height of the final output image.
         * @param width The target width of the final output image.
         * @param fill_value The value to use for padding areas. Defaults to 0.
         */
        CropAndPad(int top, int left, int height, int width, float fill_value = 0.0f);

        /**
         * @brief Executes the crop-and-pad operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting cropped and/or padded
         *         torch::Tensor of shape [C, height, width].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int top_;
        int left_;
        int height_;
        int width_;
        float fill_value_;
    };

} // namespace xt::transforms::image