#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class Sharpen
     * @brief An image transformation that sharpens an image by enhancing edges.
     *
     * This filter increases the contrast between pixels, making edges and fine
     * details appear more distinct. It works by convolving the image with a
     * sharpening kernel. This implementation uses OpenCV's 2D filter function.
     */
    class Sharpen : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a moderate sharpening effect.
         */
        Sharpen();

        /**
         * @brief Constructs the Sharpen transform.
         * @param alpha A blending factor. 0.0 means only the original image is
         *              returned, 1.0 means only the sharpened effect is returned.
         * @param lightness A factor that controls the intensity of the sharpening
         *                  kernel. Values between 0.5 and 2.0 are typical.
         */
        Sharpen(float alpha, float lightness);

        /**
         * @brief Executes the sharpening operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting sharpened torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float alpha_;
        float lightness_;
    };

} // namespace xt::transforms::image