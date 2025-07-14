#include "include/transforms/weather/particle_rain.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You would typically use this in a loop to generate an animation.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dark background image.
    torch::Tensor base_image = torch::full({3, 480, 640}, 0.1f);

    std::cout << "--- Simulating Particle Rain with Wind ---" << std::endl;

    // 2. Create the transform with a strong wind pushing rain to the right.
    xt::transforms::weather::ParticleRain rain_effect(
        3000,                                 // Particle count
        8.0f,                                 // Min speed
        15.0f,                                // Max speed
        torch::tensor({6.0f, 2.0f}),          // Wind vector (strong X, weak Y)
        10.0f,                                // Streak length
        torch::tensor({0.7, 0.75, 0.85}),     // Rain color
        42                                    // Seed
    );

    // 3. Generate a few frames of the animation.
    for (int i = 0; i < 5; ++i) {
        torch::Tensor frame = std::any_cast<torch::Tensor>(rain_effect.forward({base_image}));
        std::cout << "Generated frame " << i + 1 << "/5" << std::endl;

        // For a real test, save the frame to a file to see the angled rain.
        // std::string filename = "particle_rain_frame_" + std::to_string(i) + ".png";
        // xt::utils::save_image(frame, filename);
    }

    std::cout << "\nAnimation frames generated." << std::endl;

    return 0;
}
*/

namespace xt::transforms::weather {

    ParticleRain::ParticleRain()
            : particle_count_(1500),
              min_speed_(5.0f),
              max_speed_(10.0f),
              streak_length_(8.0f),
              seed_(0),
              generator_(torch::make_generator<torch::CPUGenerator>(0))
    {
        wind_vector_ = torch::tensor({2.0f, 1.0f}); // Gentle wind to the right-down
        rain_color_ = torch::tensor({0.8, 0.8, 0.9});
    }

    ParticleRain::ParticleRain(int particle_count, float min_speed, float max_speed, torch::Tensor wind_vector, float streak_length, torch::Tensor rain_color, int64_t seed)
            : particle_count_(particle_count),
              min_speed_(min_speed),
              max_speed_(max_speed),
              wind_vector_(std::move(wind_vector)),
              streak_length_(streak_length),
              rain_color_(std::move(rain_color)),
              seed_(seed),
              generator_(torch::make_generator<torch::CPUGenerator>(seed))
    {
        if (particle_count_ <= 0 || min_speed_ <= 0 || max_speed_ <= 0 || streak_length_ < 1) {
            throw std::invalid_argument("ParticleRain parameters must be positive.");
        }
        if (wind_vector_.numel() != 2) {
            throw std::invalid_argument("Wind vector must be a 2-element tensor (x, y).");
        }
    }

    void ParticleRain::initialize_particles(int64_t H, int64_t W) {
        auto float_opts = torch::TensorOptions().dtype(torch::kFloat32);

        // Random initial (y, x) positions
        auto y_coords = torch::rand({particle_count_}, generator_) * H;
        auto x_coords = torch::rand({particle_count_}, generator_) * W;
        particle_positions_ = torch::stack({y_coords, x_coords}, 1).to(float_opts);

        // Assign a random base falling speed to each particle
        particle_speeds_ = torch::rand({particle_count_}, generator_) * (max_speed_ - min_speed_) + min_speed_;

        is_initialized_ = true;
    }

    auto ParticleRain::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Initialization ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) { throw std::invalid_argument("ParticleRain::forward received an empty list."); }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) { throw std::invalid_argument("Input tensor is not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        auto H = image.size(1);
        auto W = image.size(2);
        if (!is_initialized_) { initialize_particles(H, W); }

        // 2. --- Calculate Velocity and Update State ---
        torch::Tensor velocity = torch::zeros_like(particle_positions_);
        velocity.select(1, 0) = particle_speeds_ + wind_vector_[1].item<float>(); // Total Y velocity
        velocity.select(1, 1) = wind_vector_[0].item<float>();                  // Total X velocity

        particle_positions_ += velocity;

        // 3. --- Reset Off-Screen Particles ---
        torch::Tensor off_screen_y = particle_positions_.select(1, 0) >= H;
        torch::Tensor off_screen_x = (particle_positions_.select(1, 1) >= W) | (particle_positions_.select(1, 1) < 0);
        torch::Tensor off_screen_mask = off_screen_y | off_screen_x;

        // Reset Y to top, give new random X
        particle_positions_.select(1, 0).masked_fill_(off_screen_mask, 0.0f);
        torch::Tensor new_x = torch::rand({particle_count_}, generator_) * W;
        particle_positions_.select(1, 1).masked_scatter_(off_screen_mask, new_x.masked_select(off_screen_mask));

        // 4. --- Render Rain Streaks ---
        torch::Tensor rain_overlay = torch::zeros({H, W}, image.options());
        torch::Tensor normalized_velocity = velocity / torch::linalg::vector_norm(velocity, 2, {1}, true);

        // Draw each streak segment by tracing backwards along the velocity vector
        for (int i = 0; i < static_cast<int>(streak_length_); ++i) {
            torch::Tensor y_coords = (particle_positions_.select(1, 0) - normalized_velocity.select(1, 0) * i).to(torch::kLong);
            torch::Tensor x_coords = (particle_positions_.select(1, 1) - normalized_velocity.select(1, 1) * i).to(torch::kLong);

            // Create a mask for valid coordinates (on-screen)
            torch::Tensor valid_mask = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W);
            auto y_idx = y_coords.masked_select(valid_mask);
            auto x_idx = x_coords.masked_select(valid_mask);

            if (y_idx.numel() > 0) {
                rain_overlay.index_put_({y_idx, x_idx}, 1.0, true);
            }
        }

        // 5. --- Composite Rain onto Image ---
        torch::Tensor rain_color_reshaped = rain_color_.to(image.options()).view({3, 1, 1});
        torch::Tensor result = image * (1.0f - rain_overlay) + rain_color_reshaped * rain_overlay;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather