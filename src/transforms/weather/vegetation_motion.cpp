#include <transforms/weather/vegetation_motion.h>

#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>

// --- Example Main (for testing) ---
// You would typically use this in a loop to generate an animation.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy image.
    torch::Tensor image = torch::zeros({3, 256, 256});
    // Let's make some vertical green "trees"
    for (int i = 40; i < 220; i += 40) {
        image.slice(2, i, i + 20).slice(0, 1, 2).fill_(0.7); // Green channel
    }

    // 2. Create a corresponding vegetation mask for the "trees".
    torch::Tensor mask = torch::zeros({1, 256, 256});
    for (int i = 40; i < 220; i += 40) {
        mask.slice(2, i, i + 20).fill_(1.0);
    }

    std::cout << "--- Simulating Vegetation Motion ---" << std::endl;

    // 3. Create the transform.
    xt::transforms::weather::VegetationMotion wind_effect(10.0f, 32.0f, 0.5f, 1337);

    // 4. Generate a few frames to see the animation.
    for (int i = 0; i < 10; ++i) {
        torch::Tensor frame = std::any_cast<torch::Tensor>(wind_effect.forward({image, mask}));
        std::cout << "Generated frame " << i + 1 << "/10" << std::endl;

        // In a real application, you would save each frame:
        // std::string filename = "vegetation_motion_frame_" + std::to_string(i) + ".png";
        // xt::utils::save_image(frame, filename);
    }

    std::cout << "\nAnimation frames generated." << std::endl;

    return 0;
}
*/

#include "include/transforms/weather/vegetation_motion.h"
#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>

// This header is required for functional calls like interpolate and grid_sample
#include <torch/nn/functional.h>

namespace xt::transforms::weather {

    VegetationMotion::VegetationMotion()
            : wind_strength_(5.0f),
              gust_scale_(32.0f),
              animation_speed_(0.5f),
              seed_(0)
    {
        generator_.set_current_seed(seed_);
    }

    VegetationMotion::VegetationMotion(float wind_strength, float gust_scale, float animation_speed, int64_t seed)
            : wind_strength_(wind_strength),
              gust_scale_(gust_scale),
              animation_speed_(animation_speed),
              seed_(seed)
    {
        generator_.set_current_seed(seed_);

        if (wind_strength_ < 0 || gust_scale_ < 1.0f || animation_speed_ < 0) {
            throw std::invalid_argument("VegetationMotion parameters must be non-negative.");
        }
    }

    void VegetationMotion::initialize_noise(int64_t H, int64_t W, torch::TensorOptions options) {
        auto low_res_H = std::max<int64_t>(1, static_cast<int64_t>(H / gust_scale_));
        auto low_res_W = std::max<int64_t>(1, static_cast<int64_t>(W / gust_scale_));

        // Generate two different low-res noise fields
        noise_a_ = torch::rand({1, 1, low_res_H, low_res_W}, generator_, options);
        noise_b_ = torch::rand({1, 1, low_res_H, low_res_W}, generator_, options);

        // --- Start of Fix ---
        // Use torch::nn::functional::interpolate for upsampling. This is the modern, correct way.
        auto interpolate_options = torch::nn::functional::InterpolateFuncOptions()
                                        .size(std::vector<int64_t>{H, W})
                                        .mode(torch::kBilinear)
                                        .align_corners(false);

        noise_a_ = torch::nn::functional::interpolate(noise_a_, interpolate_options).squeeze(0);
        noise_b_ = torch::nn::functional::interpolate(noise_b_, interpolate_options).squeeze(0);
        // --- End of Fix ---

        // Map noise from [0, 1] to [-1, 1] for left/right displacement
        noise_a_ = noise_a_ * 2.0f - 1.0f;
        noise_b_ = noise_b_ * 2.0f - 1.0f;

        is_initialized_ = true;
    }

    auto VegetationMotion::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Initialization ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("VegetationMotion requires two tensors (image and mask).");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor mask = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!image.defined() || !mask.defined()) { throw std::invalid_argument("Input tensors not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        auto H = image.size(1);
        auto W = image.size(2);
        if (!is_initialized_) { initialize_noise(H, W, image.options()); }

        // 2. --- Create Animated Displacement Field ---
        time_step_ += animation_speed_ * 0.1;
        float alpha = (sin(time_step_) + 1.0f) / 2.0f;

        torch::Tensor current_noise = torch::lerp(noise_a_, noise_b_, alpha);
        torch::Tensor displacement_field = current_noise * wind_strength_;
        displacement_field *= mask.to(image.options());

        // 3. --- Create Warping Grid for grid_sample ---
        auto x_coords = torch::linspace(-1, 1, W, image.options());
        auto y_coords = torch::linspace(-1, 1, H, image.options());
        auto mesh = torch::meshgrid({y_coords, x_coords}, "ij");
        torch::Tensor grid = torch::stack({mesh[1], mesh[0]}, 2);

        torch::Tensor normalized_displacement = displacement_field / (W / 2.0f);
        grid.select(2, 0) += normalized_displacement.squeeze(0);

        // 4. --- Apply Warping ---
        torch::Tensor warped_image = torch::nn::functional::grid_sample(
                image.unsqueeze(0),
                grid.unsqueeze(0),
                torch::nn::functional::GridSampleFuncOptions()
                    .mode(torch::kBilinear)
                    .padding_mode(torch::kZeros)
                    .align_corners(false)
        );

        return warped_image.squeeze(0);
    }

} // namespace xt::transforms::weather