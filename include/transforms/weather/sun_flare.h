#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class SunFlare
     * @brief An optical effect transform that adds a procedural sun flare to an image.
     *
     * This transform simulates a lens flare, an effect caused by a bright light
     * source (like the sun) outside the field of view or directly in it. It generates
     * several components, including a central glow, a halo, and radial streaks,
     * and additively blends them onto the input image.
     */
    class SunFlare : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Places a moderately bright, warm-colored flare in the upper-left of the frame.
         */
        SunFlare();

        /**
         * @brief Constructs the SunFlare transform with custom parameters.
         * @param sun_position A 2-element tensor `{x, y}` with normalized coordinates
         *                     (0.0 to 1.0) for the flare's center.
         * @param scale A factor to control the overall size of the flare components.
         * @param opacity The overall brightness/opacity of the flare effect [0, 1].
         * @param flare_color A 3-element tensor for the R, G, B color of the main flare.
         */
        explicit SunFlare(
                torch::Tensor sun_position,
                float scale,
                float opacity,
                torch::Tensor flare_color
        );

        /**
         * @brief Executes the sun flare generation and blending.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with the flare
         *         effect additively blended onto it.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        torch::Tensor sun_position_; // Normalized {x, y}
        float scale_;
        float opacity_;
        torch::Tensor flare_color_;
    };

} // namespace xt::transforms::weather