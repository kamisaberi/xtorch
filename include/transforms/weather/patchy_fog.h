#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class PatchyFog
     * @brief A weather transform that applies a non-uniform fog with variable density.
     *
     * This transform uses procedural noise to create a fog effect that is thicker
     * in some areas and thinner in others, simulating natural ground fog or
     * low-lying clouds. Its appearance is controlled by a base density, the
     * intensity of the patches, and the scale of the noise pattern.
     */
    class PatchyFog : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderately dense fog with visible patches.
         */
        PatchyFog();

        /**
         * @brief Constructs the PatchyFog transform with custom parameters.
         * @param base_density The minimum fog density applied everywhere, in range [0, 1].
         * @param patch_intensity The additional density applied in the thickest parts of
         *                        the fog patches. The final density is clamped to 1.0.
         * @param granularity Controls the scale of the fog patches. Smaller values (~8)
         *                    create large, rolling fog banks. Larger values (~32)
         *                    create smaller, more detailed patches.
         * @param fog_color A 3-element tensor for the R, G, B color of the fog.
         * @param seed A seed for the random number generator for reproducible patterns.
         */
        explicit PatchyFog(
                float base_density,
                float patch_intensity,
                float granularity,
                torch::Tensor fog_color,
                int64_t seed = 0
        );

        /**
         * @brief Executes the patchy fog generation and blending.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) to apply the effect to.
         * @return An std::any containing the resulting torch::Tensor with patchy fog
         *         applied, having the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float base_density_;
        float patch_intensity_;
        float granularity_;
        torch::Tensor fog_color_;
        int64_t seed_;
        torch::Generator generator_;
    };

} // namespace xt::transforms::weather