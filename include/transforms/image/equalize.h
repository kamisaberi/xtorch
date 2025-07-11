#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Equalize
     * @brief An image transformation that applies histogram equalization to improve contrast.
     *
     * This transform spreads out the most frequent intensity values in an image,
     * resulting in a higher overall contrast. For color images, it equalizes
     * the luminance (Y) channel of the YCbCr color space to avoid distorting colors.
     * This implementation uses OpenCV's equalizeHist function.
     */
    class Equalize : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         */
        Equalize();

        /**
         * @brief Executes the histogram equalization operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting equalized torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;
    };

} // namespace xt::transforms::image