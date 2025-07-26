#pragma once

#include "../common.h"

// The class is defined within your image transforms namespace
namespace xt::transforms::image {

    /**
     * @class UnderSampling
     * @brief An image transformation that extracts multiple random crops from a single image.
     *
     * This transform takes a single larger image tensor and generates `num_samples`
     * smaller, randomly positioned crops of a specified size. It uses OpenCV
     * for the underlying image manipulation. This is useful for training-time
     * data augmentation or for creating multiple views for inference.
     */
    class UnderSampling : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an uninitialized transform.
         */
        UnderSampling();

        /**
         * @brief Constructs the UnderSampling transform.
         * @param crop_size A vector of two integers {height, width} defining the
         *                  dimensions of the random crops to be extracted.
         * @param num_samples The number of random crops to generate.
         */
        UnderSampling(std::vector<int64_t> crop_size, int num_samples);

        /**
         * @brief Executes the random cropping to generate sub-samples.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting stacked torch::Tensor of
         *         shape [num_samples, C, crop_H, crop_W].
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        std::vector<int64_t> crop_size_;
        int num_samples_;
    };

} // namespace xt::transforms::image