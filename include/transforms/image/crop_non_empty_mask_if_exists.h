#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class CropNonEmptyMaskIfExists
     * @brief A transform that crops a region containing non-zero mask pixels, if a mask is provided.
     *
     * This transform has two behaviors:
     * 1. If only an image is provided, it performs a standard random crop to the target size.
     * 2. If an image and a mask are provided, it finds the bounding box of non-zero
     *    pixels in the mask and crops both the image and the mask to that region. The result
     *    is then resized to the target height and width.
     *
     * This is highly useful for segmentation tasks to ensure training crops contain the
     * object of interest.
     */
    class CropNonEmptyMaskIfExists : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        CropNonEmptyMaskIfExists();

        /**
         * @brief Constructs the transform with a target crop size.
         * @param height The target height of the final output crop.
         * @param width The target width of the final output crop.
         */
        CropNonEmptyMaskIfExists(int height, int width);

        /**
         * @brief Executes the conditional cropping operation.
         * @param tensors An initializer list that can contain:
         *                - {image}
         *                - {image, mask}
         * @return An std::any containing:
         *         - A single torch::Tensor (the cropped image) if no mask was provided.
         *         - A std::pair<torch::Tensor, torch::Tensor> (cropped image, cropped mask)
         *           if a mask was provided.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int height_;
        int width_;

        // Private helper for random cropping
        auto _random_crop(torch::Tensor& img) -> torch::Tensor;
    };

} // namespace xt::transforms::image