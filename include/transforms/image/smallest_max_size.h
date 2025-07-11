#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class SmallestMaxSize
     * @brief A transform that resizes an image so that its smallest side is equal to
     *        a specific size, while maintaining the aspect ratio.
     *
     * This is a common preprocessing step used before cropping. For example, one might
     * scale all images so their smallest side is 256px, and then perform a 224x224
     * crop.
     */
    class SmallestMaxSize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a max_size of 256.
         */
        SmallestMaxSize();

        /**
         * @brief Constructs the SmallestMaxSize transform.
         * @param max_size The target size for the smallest side of the image.
         * @param interpolation The interpolation method to use for resizing.
         *                      "linear" or "cubic" are good choices.
         */
        SmallestMaxSize(int max_size, const std::string& interpolation = "linear");

        /**
         * @brief Executes the resizing operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting resized torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_size_;
        int interpolation_flag_; // OpenCV interpolation flag
    };

} // namespace xt::transforms::image