#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class DepthBasedFog
     * @brief A weather transformation that applies fog to an image based on a depth map.
     *
     * This transform simulates atmospheric fog by blending a fog color with the
     * original image color based on the distance of each pixel from the viewer.
     * The density of the fog determines how quickly objects become obscured.
     *
     * It uses the exponential fog formula:
     * final_color = image_color * exp(-density * depth) + fog_color * (1 - exp(-density * depth))
     */
    class DepthBasedFog : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a standard gray fog with a moderate density.
         * Fog color is (0.5, 0.5, 0.5).
         * Fog density is 0.05.
         */
        DepthBasedFog();

        /**
         * @brief Constructs the DepthBasedFog transform with custom parameters.
         * @param fog_density A positive value controlling how thick the fog is. Higher
         *                    values make the fog appear more quickly with distance.
         * @param fog_color A 3-element tensor representing the R, G, B color of the
         *                  fog, with values typically in the [0, 1] range.
         */
        explicit DepthBasedFog(float fog_density, torch::Tensor fog_color);

        /**
         * @brief Executes the depth-based fog operation.
         * @param tensors An initializer list expected to contain two tensors:
         *                1. Image Tensor (C, H, W) - The visual image.
         *                2. Depth Tensor (H, W) or (1, H, W) - The depth map where each
         *                   pixel value represents the distance from the camera.
         * @return An std::any containing the resulting torch::Tensor with fog applied,
         *         having the same shape and type as the input image tensor.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float density_;
        torch::Tensor fog_color_;
    };

} // namespace xt::transforms::weather