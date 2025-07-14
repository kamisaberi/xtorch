#pragma once

#include "../common.h"

#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class StreakBasedRain
     * @brief A stateful weather transform that simulates rain using falling vertical streaks.
     *
     * This transform creates an animated rain effect by rendering short vertical lines
     * that move down the screen. The effect is stateful: each call to `forward()`
     * advances the animation by one frame. This is a common and efficient method for
     * visualizing rain in simulations and graphics.
     */
    class StreakBasedRain : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a medium-intensity rain shower with typical parameters.
         */
        StreakBasedRain();

        /**
         * @brief Constructs the StreakBasedRain transform with custom parameters.
         * @param intensity The number of raindrops/streaks to simulate.
         * @param speed The falling speed of the rain in pixels per frame.
         * @param length The visual length of the rain streaks in pixels.
         * @param rain_color A 3-element tensor for the R, G, B color of the rain streaks.
         * @param seed A seed for the random number generator for reproducibility.
         */
        explicit StreakBasedRain(
                int intensity,
                float speed,
                float length,
                torch::Tensor rain_color,
                int64_t seed = 0
        );

        /**
         * @brief Executes one step of the rain animation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W) to apply the effect to.
         * @return An std::any containing the resulting torch::Tensor with rain streaks
         *         overlaid. The internal state of the drop positions is updated.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Initializes or resets the state of all raindrops.
         * @param H The height of the image frame.
         * @param W The width of the image frame.
         */
        void initialize_drops(int64_t H, int64_t W);

        // Parameters
        int intensity_;
        float speed_;
        float length_;
        torch::Tensor rain_color_;

        // State
        int64_t seed_;
        torch::Generator generator_;
        torch::Tensor drop_positions_; // (N, 2) tensor for (y, x) coords of streak bottom
        bool is_initialized_ = false;
    };

} // namespace xt::transforms::weather