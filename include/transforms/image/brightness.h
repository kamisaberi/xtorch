#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Brightness
     * @brief An image transformation that adjusts the brightness of an image.
     *
     * This transform adds a constant value (`delta`) to every pixel in the image.
     * The result is then clamped to the valid range [0, 1] to ensure pixel
     * values are not out of bounds. A positive delta increases brightness, and a
     * negative delta decreases it.
     */
    class Brightness : public xt::Module {
    public:
        /**
         * @brief Default constructor. Creates an identity transform (delta = 0).
         */
        Brightness();

        /**
         * @brief Constructs the Brightness transform with a specific delta.
         * @param delta The value to add to each pixel. The valid range for the
         *              input image is assumed to be [0, 1], so delta should
         *              typically be in the range [-1.0, 1.0].
         */
        explicit Brightness(float delta);

        /**
         * @brief Executes the brightness adjustment.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting brightness-adjusted torch::Tensor
         *         with the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float delta_;
    };

} // namespace xt::transforms::image