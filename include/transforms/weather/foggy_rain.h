#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class FoggyRain
     * @brief A stateful transform that applies a combined fog and animated rain effect.
     *
     * This transform simulates a foggy, rainy day by first blending a uniform fog
     * layer over the image and then overlaying animated rain streaks.
     * The effect is stateful: each call to `forward()` advances the rain animation.
     */
    class FoggyRain : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderately dense gray fog with a medium intensity rain shower.
         */
        FoggyRain();

        /**
         * @brief Constructs the FoggyRain transform with custom parameters.
         * @param fog_density The opacity of the fog layer, in the range [0, 1].
         * @param fog_color A 3-element tensor for the R, G, B color of the fog.
         * @param rain_intensity The number of raindrops to simulate.
         * @param rain_speed The falling speed of the rain in pixels per frame.
         * @param rain_length The length of the rain streaks in pixels.
         * @param rain_color A 3-element tensor for the R, G, B color of the rain.
         * @param seed A seed for the random number generator for reproducibility.
         */
        explicit FoggyRain(
                float fog_density,
                torch::Tensor fog_color,
                int rain_intensity,
                float rain_speed,
                float rain_length,
                torch::Tensor rain_color,
                int64_t seed = 0
        );

        /**
         * @brief Executes one step of the fog and rain simulation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) to apply the effect to.
         * @return An std::any containing the resulting torch::Tensor with the effect
         *         applied. The internal state of the raindrop positions is updated.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Initializes or resets the state of all raindrops.
         * @param H The height of the image frame.
         * @param W The width of the image frame.
         */
        void initialize_rain(int64_t H, int64_t W);

        // Fog parameters
        float fog_density_;
        torch::Tensor fog_color_;

        // Rain parameters
        int rain_intensity_;
        float rain_speed_;
        float rain_length_;
        torch::Tensor rain_color_;

        // State
        int64_t seed_;
        torch::Generator generator_;
        torch::Tensor rain_positions_; // (N, 2) tensor for (y, x) coords
        bool is_initialized_ = false;
    };

} // namespace xt::transforms::weather