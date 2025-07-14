#include "include/transforms/weather/dynamic_shadows.h"

#include <stdexcept>
#define _USE_MATH_DEFINES
#include <math.h>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>

int main() {
    // 1. Create a dummy image (e.g., a flat green field).
    torch::Tensor image = torch::zeros({3, 128, 128});
    image[1] = 0.8; // Green channel

    // 2. Create a height map with a few "towers" that will cast shadows.
    torch::Tensor height_map = torch::zeros({128, 128});
    height_map.slice(0, 40, 60).slice(1, 40, 60).fill_(0.5);   // A square tower
    height_map.slice(0, 80, 90).slice(1, 90, 100).fill_(1.0); // A taller, smaller tower

    std::cout << "--- Applying Dynamic Shadows ---" << std::endl;
    std::cout << "Simulating light from 45 degrees (top-left)." << std::endl;

    // 3. Create the transform.
    xt::transforms::weather::DynamicShadows shadow_caster(45.0f, 50.0f, 0.6f);

    // 4. Apply the transform.
    torch::Tensor result_image = std::any_cast<torch::Tensor>(shadow_caster.forward({image, height_map}));

    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << result_image.sizes() << std::endl;
    std::cout << "The resulting image will show the green field with dark shadows cast from the towers" << std::endl;
    std::cout << "towards the bottom-right." << std::endl;

    // For a real test, you would save the images to see the result:
    // xt::utils::save_image(image, "original_field.png");
    // xt::utils::save_image(result_image, "field_with_shadows.png");
    // xt::utils::save_image(height_map, "height_map.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    DynamicShadows::DynamicShadows()
            : light_angle_degrees_(45.0f), shadow_length_(30.0f), shadow_darkness_(0.5f) {}

    DynamicShadows::DynamicShadows(float light_angle_degrees, float shadow_length, float shadow_darkness)
            : light_angle_degrees_(light_angle_degrees),
              shadow_length_(shadow_length),
              shadow_darkness_(shadow_darkness) {

        if (shadow_length_ < 0) {
            throw std::invalid_argument("Shadow length cannot be negative.");
        }
        if (shadow_darkness_ < 0.0f || shadow_darkness_ > 1.0f) {
            throw std::invalid_argument("Shadow darkness must be between 0.0 and 1.0.");
        }
    }

    auto DynamicShadows::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("DynamicShadows::forward expects two tensors (image and height map).");
        }

        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor height_map = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!image.defined() || !height_map.defined()) {
            throw std::invalid_argument("Input tensors passed to DynamicShadows are not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        // Prepare height map: ensure it has a channel dim and matches image H, W
        torch::Tensor height_map_processed = height_map.to(image.options());
        if (height_map_processed.dim() == 2) {
            height_map_processed = height_map_processed.unsqueeze(0); // (H, W) -> (1, H, W)
        }
        if (height_map_processed.size(1) != image.size(1) || height_map_processed.size(2) != image.size(2)) {
            throw std::invalid_argument("Height map dimensions must match image H and W.");
        }

        // 2. --- Shadow Map Generation ---
        // Convert angle to a direction vector for shadow casting (opposite of light)
        float light_angle_rad = (light_angle_degrees_ + 180.0f) * M_PI / 180.0f;
        float dx = cos(light_angle_rad);
        float dy = -sin(light_angle_rad); // Negative because tensor row indices increase downwards

        // Initialize maps
        torch::Tensor shadow_map = torch::ones_like(height_map_processed);
        torch::Tensor current_occluder_height = height_map_processed.clone();
        float shadow_value = 1.0f - shadow_darkness_;

        // Iteratively cast shadows
        for (int i = 0; i < static_cast<int>(shadow_length_); ++i) {
            // Shift the occluder map one step in the direction of the shadow
            current_occluder_height = torch::roll(current_occluder_height, {static_cast<int64_t>(round(dy*i)), static_cast<int64_t>(round(dx*i))}, {1, 2});

            // Where the shifted map is higher than the original terrain, a shadow is cast
            torch::Tensor is_in_shadow = current_occluder_height > height_map_processed;

            // Update the shadow map, taking the darkest shadow value found so far
            shadow_map.masked_fill_(is_in_shadow, shadow_value);
        }


        // 3. --- Apply Shadows to Image ---
        // Multiply the image by the shadow map to apply the darkening effect.
        // The shadow map (1, H, W) will broadcast correctly over the image (C, H, W).
        return image * shadow_map;
    }

} // namespace xt::transforms::weather