#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class RandomResizedCrop
     * @brief Crops a random portion of an image and resizes it to a given size.
     *
     * This is a standard augmentation for training image classification models.
     * It first crops a random region of the image with a random area (within a
     * specified scale range) and a random aspect ratio (within a specified range).
     * The cropped region is then resized to the target output size.
     */
    class RandomResizedCrop : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates a 224x224 crop, a common size for ImageNet models.
         */
        RandomResizedCrop();

        /**
         * @brief Constructs the RandomResizedCrop transform.
         *
         * @param size A pair `{height, width}` for the final output size.
         * @param scale A pair `{min, max}` specifying the range of the area to crop
         *              relative to the original image area.
         * @param ratio A pair `{min, max}` specifying the range of the aspect ratio
         *              (width / height) of the crop.
         * @param interpolation The interpolation method for the final resize.
         *                      Supported: "bilinear" (default), "nearest", "bicubic".
         */
        explicit RandomResizedCrop(
            std::pair<int, int> size,
            std::pair<double, double> scale = {0.08, 1.0},
            std::pair<double, double> ratio = {0.75, 4.0 / 3.0},
            const std::string& interpolation = "bilinear"
        );

        /**
         * @brief Executes the random resized crop.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor of the specified size.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        // Helper to get the crop parameters (top, left, height, width)
        static std::tuple<int, int, int, int> get_params(
            const torch::Tensor& img,
            const std::pair<double, double>& scale,
            const std::pair<double, double>& ratio,
            std::mt19937& gen
        );

        std::pair<int, int> size_;
        std::pair<double, double> scale_;
        std::pair<double, double> ratio_;
        std::string interpolation_mode_str_; // Store as a string
        std::mt19937 gen_;
    };

} // namespace xt::transforms::image