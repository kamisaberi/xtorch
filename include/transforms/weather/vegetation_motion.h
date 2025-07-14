#pragma once

#include "../common.h"

#include <torch/torch.h>

#pragma once

#include "../common.h"
#include <torch/torch.h>

namespace xt::transforms::weather {

    /**
     * @class VegetationMotion
     * @brief A stateful transform that simulates wind motion on vegetation.
     *
     * This transform applies a horizontal warping effect to an image based on a
     * vegetation mask, simulating the swaying of trees and grass in the wind. It uses
     * procedural noise to create a natural-looking, animated motion of wind gusts.
     * The animation is continuous and loops smoothly.
     */
    class VegetationMotion : public xt::Module {
    public:
        /**
         * @brief Default constructor.
         * Creates a moderate wind effect.
         */
        VegetationMotion();

        /**
         * @brief Constructs the VegetationMotion transform with custom parameters.
         * @param wind_strength The maximum horizontal displacement in pixels.
         * @param gust_scale Controls the size of the wind gusts. Smaller values
         *                   create larger, rolling waves; larger values create
         *                   smaller, faster ripples.
         * @param animation_speed Controls how fast the wind pattern evolves.
         * @param seed A seed for the random number generator for reproducibility.
         */
        explicit VegetationMotion(
            float wind_strength,
            float gust_scale,
            float animation_speed,
            int64_t seed = 0
        );

        /**
         * @brief Executes one step of the wind animation.
         * @param tensors An initializer list expected to contain two tensors:
         *                1. Image Tensor (C, H, W) - The source image.
         *                2. Vegetation Mask (1, H, W) or (H, W) - A grayscale mask where
         *                   white areas are warped and black areas remain static.
         * @return An std::any containing the resulting warped torch::Tensor, having the
         *         same shape and type as the input image.
         */
        auto forward(std::initializer_list<std::any> tensors) -> std::any override;

    private:
        /**
         * @brief Initializes the two noise fields used for looping animation.
         * @param H The height of the image frame.
         * @param W The width of the image frame.
         * @param options The torch::TensorOptions to use for the noise tensors.
         */
        void initialize_noise(int64_t H, int64_t W, torch::TensorOptions options);

        // Parameters
        float wind_strength_;
        float gust_scale_;
        float animation_speed_;

        // State
        int64_t seed_;
        torch::Generator generator_; // The generator is simply declared here
        double time_step_ = 0.0;
        bool is_initialized_ = false;

        // Noise fields for smooth, looping animation
        torch::Tensor noise_a_;
        torch::Tensor noise_b_;
    };

} // namespace xt::transforms::weather