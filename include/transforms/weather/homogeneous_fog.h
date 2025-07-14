#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class HomogeneousFog
     * @brief A weather transformation that applies a uniform layer of fog across the entire image.
     *
     * This transform simulates a baseline atmospheric haze or fog by linearly blending
     * the entire image with a chosen fog color. The strength of the effect is
     * controlled by a single density parameter.
     *
     * The blending formula is:
     * `final_color = image_color * (1 - density) + fog_color * density`
     */
    class HomogeneousFog : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a light-gray fog with a moderate density.
         * - Density: 0.4
         * - Color: (0.7, 0.7, 0.7)
         */
        HomogeneousFog();

        /**
         * @brief Constructs the HomogeneousFog transform with custom parameters.
         * @param density The opacity of the fog layer, must be in the range [0, 1].
         *                A value of 0 means no fog, and 1 means the image is
         *                completely replaced by the fog color.
         * @param fog_color A 3-element tensor representing the R, G, B color of the
         *                  fog, with values typically in the [0, 1] range.
         */
        explicit HomogeneousFog(float density, torch::Tensor fog_color);

        /**
         * @brief Executes the homogeneous fog operation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with uniform fog
         *         applied, having the same shape and type as the input image.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float density_;
        torch::Tensor fog_color_;
    };

} // namespace xt::transforms::weather