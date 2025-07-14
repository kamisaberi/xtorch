#include "include/transforms/weather/streak_based_rain.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You would typically use this in a loop to generate an animation.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dark, moody background image.
    torch::Tensor base_image = torch::full({3, 480, 640}, 0.2f);
    base_image[2] *= 1.5; // Make it a bit blue

    std::cout << "--- Simulating Streak-Based Rain ---" << std::endl;

    // 2. Create the transform.
    xt::transforms::weather::StreakBasedRain rain_effect(
        5000,                                   // High intensity
        20.0f,                                  // Fast speed
        6.0f,                                   // Short streaks
        torch::tensor({0.8, 0.8, 0.9}),         // Bright, slightly blue rain color
        2024                                    // Seed
    );

    // 3. Generate a few frames to see the animation.
    for (int i = 0; i < 5; ++i) {
        torch::Tensor frame = std::any_cast<torch::Tensor>(rain_effect.forward({base_image}));
        std::cout << "Generated frame " << i + 1 << "/5" << std::endl;

        // In a real application, you would save each frame:
        // std::string filename = "streak_rain_frame_" + std::to_string(i) + ".png";
        // xt::utils::save_image(frame, filename);
    }

    std::cout << "\nAnimation frames generated." << std::endl;

    return 0;
}
*/

#include "include/transforms/weather/streak_based_rain.h"
#include <stdexcept>

namespace xt::transforms::weather {

    // --- FIX 1: Corrected Generator Initialization ---
    StreakBasedRain::StreakBasedRain()
            : intensity_(2500),
              speed_(12.0f),
              length_(5.0f),
              seed_(0)
    {
        generator_.set_current_seed(seed_);
        rain_color_ = torch::tensor({0.8, 0.8, 0.9});
    }

    StreakBasedRain::StreakBasedRain(int intensity, float speed, float length, torch::Tensor rain_color, int64_t seed)
            : intensity_(intensity),
              speed_(speed),
              length_(length),
              rain_color_(std::move(rain_color)),
              seed_(seed)
    {
        generator_.set_current_seed(seed_);

        if (intensity_ <= 0 || speed_ <= 0 || length_ < 1) {
            throw std::invalid_argument("Rain intensity, speed, and length must be positive.");
        }
        if (rain_color_.numel() != 3) {
            throw std::invalid_argument("Rain color must be a 3-element tensor (R, G, B).");
        }
    }

    void StreakBasedRain::initialize_drops(int64_t H, int64_t W) {
        auto y_coords = torch::rand({intensity_}, generator_) * H;
        auto x_coords = torch::rand({intensity_}, generator_) * W;
        drop_positions_ = torch::stack({y_coords, x_coords}, 1).to(torch::kFloat32);
        is_initialized_ = true;
    }

    auto StreakBasedRain::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Initialization ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("StreakBasedRain::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) { throw std::invalid_argument("Input tensor is not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        auto H = image.size(1);
        auto W = image.size(2);
        if (!is_initialized_ || drop_positions_.size(0) != intensity_) {
            initialize_drops(H, W);
        }

        // 2. --- Update Drop State ---
        drop_positions_.select(1, 0) += speed_; // Move drops down

        torch::Tensor off_screen_mask = drop_positions_.select(1, 0) >= H;
        drop_positions_.select(1, 0).masked_fill_(off_screen_mask, 0.0f);
        torch::Tensor new_x = torch::rand({intensity_}, generator_) * W;
        drop_positions_.select(1, 1).masked_scatter_(off_screen_mask, new_x.masked_select(off_screen_mask));

        // 3. --- Render Streaks to an Overlay Mask ---
        torch::Tensor rain_overlay = torch::zeros({H, W}, image.options());

        // --- FIX 2: Corrected index_put_ Value ---
        // Create a scalar tensor containing 1.0 with the correct data type.
        torch::Tensor streak_value = torch::tensor(1.0, image.options());

        for (int i = 0; i < static_cast<int>(length_); ++i) {
            torch::Tensor y_coords = (drop_positions_.select(1, 0) - i).to(torch::kLong);
            torch::Tensor x_coords = drop_positions_.select(1, 1).to(torch::kLong);

            torch::Tensor valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W);
            auto y_idx = y_coords.masked_select(valid_mask);
            auto x_idx = x_coords.masked_select(valid_mask);

            if (y_idx.numel() > 0) {
                // Use the tensor `streak_value` and the correct two-argument overload.
                rain_overlay.index_put_({y_idx, x_idx}, streak_value);
            }
        }

        // 4. --- Composite Rain onto the Image ---
        torch::Tensor rain_color_reshaped = rain_color_.to(image.options()).view({3, 1, 1});
        torch::Tensor result = image * (1.0f - rain_overlay) + rain_color_reshaped * rain_overlay;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather