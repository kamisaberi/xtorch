#pragma once

#include "../common.h"


namespace xt::transforms::image {

    /**
     * @class ZoomBlur
     * @brief An image transformation that applies a radial "zoom" blur effect.
     *
     * This filter simulates the visual effect of zooming a camera during an
     * exposure. It works by blending multiple scaled versions of the image.
     * This implementation uses OpenCV for the underlying computation.
     */
    class ZoomBlur : public xt::Module {
    public:
        /**
         * @brief Default constructor. Uses a moderate zoom factor.
         */
        ZoomBlur();

        /**
         * @brief Constructs the ZoomBlur transform.
         * @param max_zoom The maximum zoom factor to use for creating the blur.
         *                 Values between 1.0 and 3.0 are typical.
         * @param num_steps The number of scaled images to blend together. More
         *                  steps create a smoother blur.
         */
        ZoomBlur(float max_zoom, int num_steps = 5);

        /**
         * @brief Executes the zoom blur operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting blurred torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float max_zoom_;
        int num_steps_;
    };

} // namespace xt::transforms::image