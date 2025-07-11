#pragma once

#include "../common.h"

namespace xt::transforms::image {

    /**
     * @class Emboss
     * @brief An image transformation that applies an emboss effect.
     *
     * This filter gives an image a 3D-like, carved or stamped appearance by
     * convolving it with a special kernel that highlights edges. This implementation
     * uses OpenCV's 2D filter function.
     */
    class Emboss : public xt::Module {
    public:
        /**
         * @brief Default constructor. Applies a moderate emboss effect.
         */
        Emboss();

        /**
         * @brief Constructs the Emboss transform.
         * @param alpha A blending factor. 0.0 means only the original image is
         *              returned, 1.0 means only the embossed effect is returned.
         * @param strength A factor to multiply the emboss kernel by, controlling
         *                 the intensity of the effect.
         */
        Emboss(float alpha, float strength);

        /**
         * @brief Executes the emboss filtering operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting embossed torch::Tensor with
         *         the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float alpha_;
        float strength_;
    };

} // namespace xt::transforms::image