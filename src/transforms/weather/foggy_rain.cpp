#include "include/transforms/weather/foggy_rain.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You would typically use this in a loop to generate an animation.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy background image.
    torch::Tensor base_image = torch::zeros({3, 480, 640});
    base_image[1].fill_(0.2); // Dark green
    base_image[0].fill_(0.1);

    std::cout << "--- Simulating Foggy Rain ---" << std::endl;

    // 2. Create the transform with custom parameters.
    xt::transforms::weather::FoggyRain foggy_rain_effect(
        0.4f,                                     // Fog density
        torch::tensor({0.6, 0.6, 0.65}),          // Fog color (light gray-blue)
        4000,                                     // Rain intensity
        15.0f,                                    // Rain speed
        5.0f,                                     // Rain streak length
        torch::tensor({0.8, 0.8, 0.9}),           // Rain color (bright, slightly blue)
        123                                       // Seed
    );

    // 3. Loop to simulate animation frames.
    int num_frames = 10;
    for (int i = 0; i < num_frames; ++i) {
        torch::Tensor frame = std::any_cast<torch::Tensor>(foggy_rain_effect.forward({base_image}));
        std::cout << "Generated frame " << i + 1 << "/" << num_frames << std::endl;

        // In a real application, you would save each frame:
        // std::string filename = "foggy_rain_frame_" + std::to_string(i) + ".png";
        // xt::utils::save_image(frame, filename);
    }

    std::cout << "\nAnimation frames generated." << std::endl;

    return 0;
}
*/

namespace xt::transforms::weather {

    FoggyRain::FoggyRain()
            : fog_density_(0.3f),
              rain_intensity_(2000),
              rain_speed_(10.0f),
              rain_length_(4.0f),
              seed_(0),
              generator_(torch::make_generator<torch::CPUGenerator>(0))
    {
        fog_color_ = torch::tensor({0.5, 0.5, 0.5});
        rain_color_ = torch::tensor({0.7, 0.7, 0.8});
    }

    FoggyRain::FoggyRain(float fog_density, torch::Tensor fog_color, int rain_intensity, float rain_speed, float rain_length, torch::Tensor rain_color, int64_t seed)
            : fog_density_(fog_density),
              fog_color_(std::move(fog_color)),
              rain_intensity_(rain_intensity),
              rain_speed_(rain_speed),
              rain_length_(rain_length),
              rain_color_(std::move(rain_color)),
              seed_(seed),
              generator_(torch::make_generator<torch::CPUGenerator>(seed))
    {
        if (fog_density_ < 0 || rain_intensity_ <= 0 || rain_speed_ <= 0 || rain_length_ < 1) {
            throw std::invalid_argument("Fog and rain parameters must be positive.");
        }
    }

    void FoggyRain::initialize_rain(int64_t H, int64_t W) {
        // Random initial (y, x) positions for each raindrop
        auto y_coords = torch::rand({rain_intensity_}, generator_) * H;
        auto x_coords = torch::rand({rain_intensity_}, generator_) * W;
        rain_positions_ = torch::stack({y_coords, x_coords}, 1).to(torch::kFloat32);
        is_initialized_ = true;
    }

    auto FoggyRain::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Initialization ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("FoggyRain::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) { throw std::invalid_argument("Input tensor is not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        auto H = image.size(1);
        auto W = image.size(2);

        if (!is_initialized_ || rain_positions_.size(0) != rain_intensity_) {
            initialize_rain(H, W);
        }

        // 2. --- Apply Uniform Fog Layer ---
        torch::Tensor fog_color_reshaped = fog_color_.to(image.options()).view({3, 1, 1});
        torch::Tensor foggy_image = image * (1.0f - fog_density_) + fog_color_reshaped * fog_density_;

        // 3. --- Update Rain State ---
        rain_positions_.select(1, 0) += rain_speed_; // Move drops down
        torch::Tensor off_screen_mask = rain_positions_.select(1, 0) >= H;
        rain_positions_.select(1, 0).masked_fill_(off_screen_mask, 0.0f); // Reset Y to top
        torch::Tensor new_x = torch::rand({rain_intensity_}, generator_) * W;
        rain_positions_.select(1, 1).masked_scatter_(off_screen_mask, new_x.masked_select(off_screen_mask));

        // 4. --- Render Rain Streaks ---
        torch::Tensor rain_overlay = torch::zeros({H, W}, image.options());
        for (int i = 0; i < static_cast<int>(rain_length_); ++i) {
            // Calculate coordinates for each segment of the streak
            torch::Tensor y_coords = (rain_positions_.select(1, 0) - i).to(torch::kLong);
            torch::Tensor x_coords = rain_positions_.select(1, 1).to(torch::kLong);

            // Create a mask for valid coordinates (on-screen)
            torch::Tensor valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W);

            // Get the valid indices
            auto y_idx = y_coords.masked_select(valid_mask);
            auto x_idx = x_coords.masked_select(valid_mask);

            // Draw the streak segments onto the overlay
            if (y_idx.numel() > 0) {
                rain_overlay.index_put_({y_idx, x_idx}, 1.0, true);
            }
        }

        // 5. --- Composite Rain onto Foggy Image ---
        torch::Tensor rain_color_reshaped = rain_color_.to(image.options()).view({3, 1, 1});
        torch::Tensor result = foggy_image * (1.0f - rain_overlay) + rain_color_reshaped * rain_overlay;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather