#include "include/transforms/weather/falling_snow.h"
#include <stdexcept>

// --- Example Main (for testing) ---
// You would typically use this in a loop to generate an animation.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Load a base image or create a dummy one.
    torch::Tensor base_image = torch::zeros({3, 480, 640});
    base_image[0].fill_(0.1); // Dark blueish background
    base_image[2].fill_(0.3);

    std::cout << "--- Simulating Falling Snow ---" << std::endl;

    // 2. Create the transform.
    xt::transforms::weather::FallingSnow snow_effect(2000, 1.0, 5.0, 0.4, 1.0, torch::tensor({1.0, 1.0, 1.0}));

    // 3. Loop to simulate animation frames.
    int num_frames = 10;
    for (int i = 0; i < num_frames; ++i) {
        // Each call to forward() updates the snow positions and applies it to the image.
        torch::Tensor frame = std::any_cast<torch::Tensor>(snow_effect.forward({base_image}));
        std::cout << "Generated frame " << i + 1 << "/" << num_frames << std::endl;

        // In a real application, you would save each frame:
        // std::string filename = "snow_frame_" + std::to_string(i) + ".png";
        // xt::utils::save_image(frame, filename);
    }

    std::cout << "\nAnimation frames generated." << std::endl;

    return 0;
}
*/

namespace xt::transforms::weather {

    // --- Start of Fix ---
    FallingSnow::FallingSnow()
            : flake_count_(1000),
              min_speed_(1.0f),
              max_speed_(4.0f),
              min_opacity_(0.3f),
              max_opacity_(0.8f),
              seed_(0)
              // The generator_ member is implicitly default-constructed here.
    {
        generator_.set_current_seed(seed_); // Set the seed using the public API.
        snow_color_ = torch::tensor({1.0, 1.0, 1.0});
    }

    FallingSnow::FallingSnow(int flake_count, float min_speed, float max_speed, float min_opacity, float max_opacity, torch::Tensor snow_color, int64_t seed)
            : flake_count_(flake_count),
              min_speed_(min_speed),
              max_speed_(max_speed),
              min_opacity_(min_opacity),
              max_opacity_(max_opacity),
              snow_color_(std::move(snow_color)),
              seed_(seed)
              // The generator_ member is implicitly default-constructed here.
    {
        generator_.set_current_seed(seed_); // Set the seed using the public API.

        if (flake_count_ <= 0 || min_speed_ <= 0 || max_speed_ <= 0 || min_opacity_ < 0 || max_opacity_ < 0) {
            throw std::invalid_argument("Snow parameters must be positive.");
        }
        if (min_speed_ > max_speed_ || min_opacity_ > max_opacity_) {
            throw std::invalid_argument("Min values cannot be greater than max values.");
        }
    }
    // --- End of Fix ---

    void FallingSnow::initialize_flakes(int64_t H, int64_t W) {
        auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);

        // Random initial Y positions across the screen height
        auto y_coords = torch::rand({flake_count_}, generator_) * H;
        // Random initial X positions across the screen width
        auto x_coords = torch::rand({flake_count_}, generator_) * W;

        // Combine into a (N, 2) tensor
        flake_positions_ = torch::stack({y_coords, x_coords}, 1).to(float_opts);

        // Assign a random speed and opacity to each flake
        flake_speeds_ = torch::rand({flake_count_}, generator_) * (max_speed_ - min_speed_) + min_speed_;
        flake_opacities_ = torch::rand({flake_count_}, generator_) * (max_opacity_ - min_opacity_) + min_opacity_;

        is_initialized_ = true;
    }


    auto FallingSnow::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Initialization ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("FallingSnow::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to FallingSnow is not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        auto H = image.size(1);
        auto W = image.size(2);

        // Initialize flakes on the first run or if image size changes
        // Corrected the condition to check against the actual image dimensions
        if (!is_initialized_ || flake_positions_.sizes()[0] != flake_count_) {
             initialize_flakes(H, W);
        }

        // 2. --- Update Flake State ---
        // Move flakes downwards according to their speed
        flake_positions_.select(1, 0) += flake_speeds_;

        // 3. --- Handle Flakes Leaving the Screen ---
        // Find flakes that have moved past the bottom edge (y >= H)
        torch::Tensor off_screen_mask = flake_positions_.select(1, 0) >= H;

        // For each off-screen flake, reset its Y position to the top (y=0)
        flake_positions_.select(1, 0).masked_fill_(off_screen_mask, 0.0f);

        // And give it a new random X position
        torch::Tensor new_x_coords = torch::rand({flake_count_}, generator_) * W;
        flake_positions_.select(1, 1).masked_scatter_(off_screen_mask, new_x_coords.masked_select(off_screen_mask));

        // 4. --- Draw Flakes onto a Mask ---
        // Create a transparent mask
        torch::Tensor snow_mask = torch::zeros({H, W}, image.options());

        // Get integer coordinates for indexing
        torch::Tensor y_idx = flake_positions_.select(1, 0).to(torch::kLong);
        torch::Tensor x_idx = flake_positions_.select(1, 1).to(torch::kLong);

        // Clamp coordinates to be within image bounds
        y_idx.clamp_max_(H - 1);
        x_idx.clamp_max_(W - 1);

        // Place flakes on the mask using their opacity. This is a simple but effective method.
        // A more advanced approach would draw blurred circles (sprites).
        snow_mask.index_put_({y_idx, x_idx}, flake_opacities_, true);

        // 5. --- Blend Image with Snow Mask ---
        torch::Tensor snow_color_reshaped = snow_color_.to(image.options()).view({3, 1, 1});
        torch::Tensor result = image * (1.0f - snow_mask) + snow_color_reshaped * snow_mask;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather