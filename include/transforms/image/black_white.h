#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class BlackWhite
     * @brief An image transformation that applies binary thresholding to create a
     *        pure black-and-white image.
     *
     * This transform converts an input image into a binary image where pixel values
     * are either 0 or 1. If the input is a color image, it is first converted to
     * grayscale. Then, any pixel with an intensity greater than the specified
     * threshold becomes 1 (white), and all others become 0 (black).
     */
    class BlackWhite : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a default threshold of 0.5.
         */
        BlackWhite();

        /**
         * @brief Constructs the transform with a specific threshold.
         * @param threshold The intensity value (typically between 0.0 and 1.0)
         *                  to use as the cutoff.
         */
        explicit BlackWhite(float threshold);

        /**
         * @brief Executes the binary thresholding operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W). Can be 1-channel or 3-channel.
         * @return An std::any containing the resulting binary torch::Tensor of
         *         shape [1, H, W] with values of 0 or 1.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float threshold_;
    };

} // namespace xt::transforms::image