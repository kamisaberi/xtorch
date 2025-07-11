#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class PixelDropout
     * @brief An image transformation that randomly sets a fraction of pixels to a given value.
     *
     * This is a simple form of dropout applied directly to the image pixels,
     * which can help improve model robustness. It is similar to adding
     * salt-and-pepper noise.
     */
    class PixelDropout : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a dropout probability of 0.05.
         */
        PixelDropout();

        /**
         * @brief Constructs the PixelDropout transform.
         * @param p The probability of a pixel being dropped (set to the drop value).
         *          Should be in the range [0, 1].
         * @param drop_value The value to set the dropped pixels to. Defaults to 0.
         */
        explicit PixelDropout(double p, float drop_value = 0.0f);

        /**
         * @brief Executes the pixel dropout operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with some pixels
         *         dropped. The shape and type remain the same.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        double p_;
        float drop_value_;
    };

} // namespace xt::transforms::image