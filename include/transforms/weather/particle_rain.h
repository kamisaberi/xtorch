#pragma once

#include "../common.h"


#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class ParticleRain
     * @brief A stateful transform that simulates rain as particles influenced by wind.
     *
     * This transform creates a more dynamic rain effect by treating each drop as a
     * particle with a velocity vector. A global wind force can be applied to make
     * the rain fall at an angle. The effect is stateful, with each `forward()` call
     * advancing the simulation.
     */
    class ParticleRain : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderate rain shower with a slight wind from the left.
         */
        ParticleRain();

        /**
         * @brief Constructs the ParticleRain transform with custom physics.
         * @param particle_count The number of raindrops to simulate.
         * @param min_speed The minimum base falling speed (gravity).
         * @param max_speed The maximum base falling speed.
         * @param wind_vector A 2-element tensor for the (x, y) wind components.
         *                    Positive x pushes rain right, positive y pushes it down.
         * @param streak_length The visual length of the rain streaks in pixels.
         * @param rain_color A 3-element tensor for the R, G, B color of the rain.
         * @param seed A seed for the random number generator.
         */
        explicit ParticleRain(
                int particle_count,
                float min_speed,
                float max_speed,
                torch::Tensor wind_vector,
                float streak_length,
                torch::Tensor rain_color,
                int64_t seed = 0
        );

        /**
         * @brief Executes one step of the particle rain simulation.
         * @param tensors An initializer list expected to contain a single 3D image
         *                tensor (C, H, W).
         * @return An std::any containing the resulting torch::Tensor with the rain
         *         effect. The internal state of the particle positions is updated.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Initializes or resets the state of all rain particles.
         * @param H The height of the image frame.
         * @param W The width of the image frame.
         */
        void initialize_particles(int64_t H, int64_t W);

        // Parameters
        int particle_count_;
        float min_speed_;
        float max_speed_;
        torch::Tensor wind_vector_; // (x, y) components
        float streak_length_;
        torch::Tensor rain_color_;

        // State
        int64_t seed_;
        torch::Generator generator_;
        torch::Tensor particle_positions_; // (N, 2) tensor for (y, x) coords
        torch::Tensor particle_speeds_;    // (N, 1) tensor for base downward speed
        bool is_initialized_ = false;
    };

} // namespace xt::transforms::weather