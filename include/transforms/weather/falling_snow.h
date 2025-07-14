#pragma once

#include "../common.h"

#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class FallingSnow
     * @brief A stateful weather transformation that adds an animated falling snow effect.
     *
     * This transform simulates snow falling over an image. It is stateful: each call
     * to `forward()` advances the animation by one time step, causing the snowflakes
     * to fall. When a snowflake moves off the bottom of the screen, it reappears at
     * the top in a new random location.
     */
    class FallingSnow : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderate snow effect with 1000 flakes.
         * - Flake Count: 1000
         * - Speed: min=1.0, max=4.0 pixels/frame
         * - Opacity: min=0.3, max=0.8
         * - Color: White (1.0, 1.0, 1.0)
         * - Seed: 0
         */
        FallingSnow();

        /**
         * @brief Constructs the FallingSnow transform with custom parameters.
         * @param flake_count The number of snowflakes to simulate.
         * @param min_speed The minimum falling speed in pixels per frame.
         * @param max_speed The maximum falling speed in pixels per frame.
         * @param min_opacity The minimum opacity of a snowflake [0, 1].
         * @param max_opacity The maximum opacity of a snowflake [0, 1].
         * @param snow_color A 3-element tensor for the R, G, B color of the snow [0, 1].
         * @param seed A seed for the random number generator for reproducibility.
         */
        explicit FallingSnow(
                int flake_count,
                float min_speed,
                float max_speed,
                float min_opacity,
                float max_opacity,
                torch::Tensor snow_color,
                int64_t seed = 0
        );

        /**
         * @brief Executes one step of the snow animation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) to apply the effect to.
         * @return An std::any containing the resulting torch::Tensor with snow
         *         overlaid. The internal state of the snowflake positions is updated.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Initializes or resets the state of all snowflakes.
         * @param H The height of the image frame.
         * @param W The width of the image frame.
         */
        void initialize_flakes(int64_t H, int64_t W);

        int flake_count_;
        float min_speed_;
        float max_speed_;
        float min_opacity_;
        float max_opacity_;
        torch::Tensor snow_color_;
        int64_t seed_;
        torch::Generator generator_;

        // State tensors for snowflakes
        torch::Tensor flake_positions_; // (N, 2) tensor for (y, x) coords
        torch::Tensor flake_speeds_;    // (N, 1) tensor
        torch::Tensor flake_opacities_; // (N, 1) tensor

        bool is_initialized_ = false;
    };

} // namespace xt::transforms::weather