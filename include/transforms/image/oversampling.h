#pragma once

#include <vector>
#include <any>
#include <torch/torch.h>
#include "../common.h"

// It's good practice to only forward-declare heavy dependencies in header files
// if possible, but for templates or simple types, including is fine.
// Since we only use std::vector<int64_t>, we don't need to include all of OpenCV here.

// Assuming your base module class is defined in a file like this
// #include "include/module.h"

// The class is defined within your custom namespace structure
namespace xt::transforms::image {

    /**
     * @class OverSampling
     * @brief An image transformation that performs 10-crop test-time augmentation.
     *
     * This transform takes a single image tensor and generates 10 augmented versions:
     * one from each of the four corners, one from the center, and a horizontal flip
     * of each of these five crops. It uses OpenCV for the underlying image manipulation.
     * The output is a single stacked tensor of shape [10, C, H, W].
     */
    class OverSampling : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized transform.
         */
        OverSampling();

        /**
         * @brief Constructs the OverSampling transform with a specified crop size.
         * @param crop_size A vector of two integers {height, width} defining the
         *                  dimensions of the crops to be extracted.
         */
        explicit OverSampling(std::vector<int64_t> crop_size);

        /**
         * @brief Executes the 10-crop augmentation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting stacked torch::Tensor of
         *         shape [10, C, crop_H, crop_W].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Stores the target size for the crops, e.g., {224, 224}.
        std::vector<int64_t> crop_size;
    };

} // namespace xt::transforms::image