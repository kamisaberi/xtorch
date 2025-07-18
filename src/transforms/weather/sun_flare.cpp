#include <transforms/weather/sun_flare.h>

#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy image, e.g., a dark landscape.
    torch::Tensor image = torch::zeros({3, 512, 512});
    image.slice(0, 0, 1).slice(1, 256, 512).fill_(0.2); // Dark blue sky
    image.slice(0, 1, 2).slice(1, 0, 256).fill_(0.1); // Dark green ground

    std::cout << "--- Applying Sun Flare Effect ---" << std::endl;

    // 2. Create the transform, placing the sun in the upper right.
    xt::transforms::weather::SunFlare sun_flare_effect(
        torch::tensor({0.85f, 0.15f}),         // Position (85% right, 15% down)
        1.0f,                                 // Scale
        0.8f,                                 // Opacity
        torch::tensor({1.0f, 0.95f, 0.8f})    // Warm, yellowish-white color
    );

    // 3. Apply the transform.
    torch::Tensor result_image = std::any_cast<torch::Tensor>(sun_flare_effect.forward({image}));

    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << result_image.sizes() << std::endl;
    std::cout << "The resulting image will have a bright flare in the upper-right." << std::endl;

    // To see the effect, you would save the output image:
    // xt::utils::save_image(result_image, "sun_flare_effect.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    SunFlare::SunFlare()
            : scale_(1.0f),
              opacity_(0.6f)
    {
        sun_position_ = torch::tensor({0.2, 0.2}); // Upper-left
        flare_color_ = torch::tensor({1.0f, 0.9f, 0.7f}); // Warm white
    }

    SunFlare::SunFlare(torch::Tensor sun_position, float scale, float opacity, torch::Tensor flare_color)
            : sun_position_(std::move(sun_position)),
              scale_(scale),
              opacity_(opacity),
              flare_color_(std::move(flare_color))
    {
        if (sun_position_.numel() != 2) {
            throw std::invalid_argument("Sun position must be a 2-element tensor {x, y}.");
        }
        if (opacity_ < 0.0f || opacity_ > 1.0f) {
            throw std::invalid_argument("Opacity must be between 0.0 and 1.0.");
        }
    }

    auto SunFlare::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) { throw std::invalid_argument("SunFlare::forward received an empty list."); }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) { throw std::invalid_argument("Input tensor is not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        // 2. --- Setup ---
        auto H = image.size(1);
        auto W = image.size(2);
        auto opts = image.options();

        // Create a blank layer to draw the flare onto
        torch::Tensor flare_layer = torch::zeros_like(image);

        // Calculate absolute pixel position of the sun
        float sun_x = sun_position_[0].item<float>() * W;
        float sun_y = sun_position_[1].item<float>() * H;

        // Create coordinate grids
        auto x_coords = torch::arange(0, W, opts).view({1, -1}).expand({H, -1});
        auto y_coords = torch::arange(0, H, opts).view({-1, 1}).expand({-1, W});

        // 3. --- Generate Primary Glow ---
        // Calculate distance from every pixel to the sun position
        auto dist_from_sun = torch::sqrt(torch::pow(x_coords - sun_x, 2) + torch::pow(y_coords - sun_y, 2));

        // Create a radial glow using an inverse distance falloff
        auto glow_radius = 200.0f * scale_;
        torch::Tensor primary_glow = 1.0f / (dist_from_sun / glow_radius + 1.0f);
        primary_glow = torch::pow(primary_glow, 4.0); // Power to make the falloff sharper

        // 4. --- Generate Halo/Ghost ---
        // The halo appears reflected through the center of the image
        float center_x = W / 2.0f;
        float center_y = H / 2.0f;
        float halo_x = center_x + (center_x - sun_x);
        float halo_y = center_y + (center_y - sun_y);

        auto dist_from_halo = torch::sqrt(torch::pow(x_coords - halo_x, 2) + torch::pow(y_coords - halo_y, 2));
        auto halo_radius = 120.0f * scale_;
        torch::Tensor halo_ring = torch::exp(-torch::pow(dist_from_halo - halo_radius, 2) / (2 * pow(30 * scale_, 2)));
        halo_ring *= 0.5f; // Make halo less intense than the main glow

        // 5. --- Composite Flare Components ---
        // Combine glow and halo into a single grayscale mask
        torch::Tensor flare_mask = torch::clamp(primary_glow + halo_ring, 0.0, 1.0);

        // Colorize the mask and apply opacity
        torch::Tensor flare_color_reshaped = flare_color_.to(opts).view({3, 1, 1});
        flare_layer += flare_mask.unsqueeze(0) * flare_color_reshaped * opacity_;

        // 6. --- Additive Blending ---
        // Add the flare layer to the original image
        torch::Tensor result = image + flare_layer;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather