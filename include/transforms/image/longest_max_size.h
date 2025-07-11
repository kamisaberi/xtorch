#pragma once

#include "../common.h"



namespace xt::transforms::image {

    /**
     * @class LongestMaxSize
     * @brief A transform that resizes an image so that its longest side does not
     *        exceed a specific size, while maintaining the aspect ratio.
     *
     * If the longest side of the input image is greater than `max_size`, the image
     * is downscaled. If it is smaller or equal, the image is returned unchanged.
     * This is useful for ensuring inputs to a model do not exceed a certain
     * dimensional limit.
     */
    class LongestMaxSize : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a max_size of 1024.
         */
        LongestMaxSize();

        /**
         * @brief Constructs the LongestMaxSize transform.
         * @param max_size The maximum allowable size for the longest side of the image.
         * @param interpolation The interpolation method to use for downscaling.
         *                      "area" is recommended.
         */
        LongestMaxSize(int max_size, const std::string& interpolation = "area");

        /**
         * @brief Executes the resizing operation if necessary.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting resized (or original)
         *         torch::Tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int max_size_;
        int interpolation_flag_; // OpenCV interpolation flag
    };

} // namespace xt::transforms::image