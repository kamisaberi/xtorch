#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class RandomCropNearBbox
     * @brief Crops a region of a specified size near a given bounding box.
     *
     * This transform is useful for creating training samples that are guaranteed
     * to contain a specific object of interest. It samples a crop window whose
     * center is within a specified maximum distance from the center of the
     * provided bounding box.
     */
    class RandomCropNearBbox : public xt::Module {
    public:
        /**
         * @brief Default constructor. Does nothing as crop size is required.
         */
        RandomCropNearBbox();

        /**
         * @brief Constructs the RandomCropNearBbox transform.
         *
         * @param crop_size A pair `{height, width}` specifying the desired output size.
         * @param max_distance The maximum allowed distance (in pixels) between the
         *                     center of the crop and the center of the bounding box.
         *                     A value of 0 means the crop will be centered exactly
         *                     on the bounding box.
         */
        explicit RandomCropNearBbox(
            std::pair<int, int> crop_size,
            int max_distance = 20
        );

        /**
         * @brief Executes the random crop operation near a bounding box.
         * @param tensors An initializer list expected to contain two items:
         *                1. A 3D image tensor (C, H, W).
         *                2. A 1D tensor representing the bounding box in
         *                   [xmin, ymin, xmax, ymax] format.
         * @return An std::any containing the resulting cropped torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::pair<int, int> crop_size_;
        int max_distance_;
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image