#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class Crop
     * @brief An image transformation that extracts a rectangular region from an image.
     *
     * This transform crops the given image at a specified location and size.
     * The crop region is defined by its top-left corner coordinates and its
     * desired height and width.
     */
    class Crop : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized (and unusable) transform.
         */
        Crop();

        /**
         * @brief Constructs the Crop transform.
         * @param top The vertical coordinate of the top-left corner of the crop.
         * @param left The horizontal coordinate of the top-left corner of the crop.
         * @param height The height of the crop.
         * @param width The width of the crop.
         */
        Crop(int top, int left, int height, int width);

        /**
         * @brief Executes the cropping operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting cropped torch::Tensor of
         *         shape [C, height, width].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int top_;
        int left_;
        int height_;
        int width_;
    };

} // namespace xt::transforms::image