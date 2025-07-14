#pragma once

#include "../common.h"

#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class DustSandClouds
     * @brief A weather transformation that overlays a procedural dust or sand storm effect onto an image.
     *
     * This transform generates a turbulent, cloud-like pattern using procedural noise
     * and blends it with the input image to simulate a dust storm. The appearance can
     * be controlled by setting the dust color, its overall density, the granularity of
     * the cloud patterns, and a seed for reproducibility.
     */
    class DustSandClouds : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderately dense sand storm effect with a typical sandy-brown color.
         * - Density: 0.7
         * - Granularity: 16.0 (medium-sized cloud patterns)
         * - Color: (210, 180, 140) converted to [0, 1] range.
         * - Seed: 0
         */
        DustSandClouds();

        /**
         * @brief Constructs the DustSandClouds transform with custom parameters.
         * @param density The overall opacity of the dust cloud, in the range [0, 1].
         *                0 is fully transparent, 1 is nearly opaque in the densest parts.
         * @param granularity Controls the scale of the noise pattern. Smaller values (~4-8)
         *                    create large, rolling clouds. Larger values (~32-64) create
         *                    finer, grainier dust.
         * @param dust_color A 3-element tensor representing the R, G, B color of the
         *                   dust, with values in the [0, 1] range.
         * @param seed A seed for the random number generator to ensure the generated
         *             noise pattern is reproducible.
         */
        explicit DustSandClouds(float density, float granularity, torch::Tensor dust_color, int64_t seed = 0);

        /**
         * @brief Executes the dust storm generation and blending.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) to apply the effect to.
         * @return An std::any containing the resulting torch::Tensor with the dust
         *         effect applied, having the same shape and type as the input.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        float density_;
        float granularity_;
        torch::Tensor dust_color_;
        int64_t seed_;
    };

} // namespace xt::transforms::weather