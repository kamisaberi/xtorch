#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class MedianBlur
     * @brief An image transformation that applies a median filter.
     *
     * This is a non-linear filter that is highly effective at removing "salt-and-pepper"
     * noise while preserving edges better than linear filters like Gaussian blur.
     * This implementation uses OpenCV's medianBlur function.
     */
    class MedianBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a 3x3 kernel.
         */
        MedianBlur();

        /**
         * @brief Constructs the MedianBlur transform.
         * @param kernel_size The size of the median filter kernel. Must be an
         *                    odd integer greater than 1 (e.g., 3, 5, 7).
         */
        explicit MedianBlur(int kernel_size);

        /**
         * @brief Executes the median blur operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        int kernel_size_;
    };

} // namespace xt::transforms::image