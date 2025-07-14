#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class DynamicShadows
     * @brief A weather transformation that casts dynamic shadows on an image using a height map.
     *
     * This transform simulates the effect of a directional light source (like the sun)
     * casting shadows from taller objects or terrain features defined in a height map.
     * The shadow's direction, length, and darkness are all configurable.
     */
    class DynamicShadows : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates shadows cast from a top-left light source (45 degrees) with
         * medium length and darkness.
         * - Light Angle: 45.0 degrees
         * - Shadow Length: 30.0 units
         * - Shadow Darkness: 0.5 (50% opaque)
         */
        DynamicShadows();

        /**
         * @brief Constructs the DynamicShadows transform with custom parameters.
         * @param light_angle_degrees The angle of the light source in degrees (0-360).
         *                            0 degrees is to the right, 90 is down, etc.
         * @param shadow_length The maximum length of the shadows. This controls the number
         *                      of steps in the shadow projection algorithm.
         * @param shadow_darkness A value from 0.0 (no shadow) to 1.0 (completely black)
         *                        that controls the opacity of the shadows.
         */
        explicit DynamicShadows(float light_angle_degrees, float shadow_length, float shadow_darkness);

        /**
         * @brief Executes the shadow casting operation.
         * @param tensors An initializer list expected to contain two tensors:
         *                1. Image Tensor (C, H, W) - The visual image.
         *                2. Height Map Tensor (H, W) or (1, H, W) - A grayscale map where
         *                   brighter values represent higher elevation.
         * @return An std::any containing the resulting torch::Tensor with shadows
         *         applied, having the same shape and type as the input image.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float light_angle_degrees_;
        float shadow_length_;
        float shadow_darkness_;
    };

} // namespace xt::transforms::weather