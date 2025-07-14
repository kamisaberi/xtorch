#include "include/transforms/weather/wet_ground.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy image of a dark gray "asphalt" road.
    torch::Tensor image = torch::full({3, 256, 512}, 0.25f);

    // 2. Create a ground map. Let's make some areas have higher "puddle potential".
    torch::Tensor ground_map = torch::zeros({1, 256, 512});
    // The whole image is ground.
    ground_map.fill_(0.5);
    // Add a patch with high puddle potential (e.g., a dip in the road).
    ground_map.slice(2, 200, 400).slice(1, 100, 150).fill_(0.95);
    // Add a patch with low puddle potential (a high point).
    ground_map.slice(2, 50, 150).slice(1, 180, 220).fill_(0.1);

    std::cout << "--- Applying Wet Ground Effect ---" << std::endl;

    // 3. Create the transform.
    xt::transforms::weather::WetGround wet_effect(
        0.5f,                                     // 50% darkening
        0.2f,                                     // 20% specular sheen
        0.8f,                                     // Puddles form where map value > 0.8
        0.7f,                                     // Puddle reflections are 70% intense
        torch::tensor({0.4, 0.5, 0.7})            // Reflect a blueish-gray sky
    );

    // 4. Apply the transform.
    torch::Tensor result_image = std::any_cast<torch::Tensor>(wet_effect.forward({image, ground_map}));

    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << result_image.sizes() << std::endl;
    std::cout << "The road should now appear darker and shinier, with a blue puddle." << std::endl;

    // To see the effect, you would save the output image:
    // xt::utils::save_image(result_image, "wet_ground_effect.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    WetGround::WetGround()
            : darkening_(0.4f),
              specular_intensity_(0.15f),
              puddle_threshold_(0.75f),
              reflection_intensity_(0.6f)
    {
        reflection_color_ = torch::tensor({0.6, 0.6, 0.6}); // Default gray sky reflection
    }

    WetGround::WetGround(float darkening, float specular, float puddle_thresh, float reflect_intensity, torch::Tensor reflect_color)
            : darkening_(darkening),
              specular_intensity_(specular),
              puddle_threshold_(puddle_thresh),
              reflection_intensity_(reflect_intensity),
              reflection_color_(std::move(reflect_color))
    {
        if (darkening_ < 0 || specular_intensity_ < 0 || puddle_threshold_ < 0 || reflection_intensity_ < 0 ||
            darkening_ > 1 || specular_intensity_ > 1 || puddle_threshold_ > 1 || reflection_intensity_ > 1) {
            throw std::invalid_argument("WetGround intensity/threshold factors must be between 0.0 and 1.0.");
        }
    }

    auto WetGround::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("WetGround::forward requires two tensors (image and ground_spec_map).");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor ground_map = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!image.defined() || !ground_map.defined()) { throw std::invalid_argument("Input tensors not defined."); }
        if (image.dim() != 3) { throw std::invalid_argument("Input image must be a 3D tensor (C, H, W)."); }

        // Ensure ground_map is broadcastable to image shape
        torch::Tensor ground_map_processed = ground_map.to(image.options());
        if (ground_map_processed.dim() == 2) {
            ground_map_processed = ground_map_processed.unsqueeze(0); // (H, W) -> (1, H, W)
        }

        // 2. --- Create Masks ---
        // A mask for all ground areas (where map value is > 0)
        torch::Tensor ground_mask = ground_map_processed > 0;
        // A mask for puddle areas (where ground map value is above the threshold)
        torch::Tensor puddle_mask = (ground_map_processed > puddle_threshold_) & ground_mask;

        // 3. --- Apply Darkening and Specular Sheen ---
        // Start with the original image
        torch::Tensor result_image = image.clone();

        // Darken the ground: image * (1 - factor)
        // Add specular sheen: image + factor
        // Combined: image * (1 - darken) + specular
        float darken_multiplier = 1.0f - darkening_;
        torch::Tensor wet_effect = image * darken_multiplier + specular_intensity_;

        // Apply this effect only on the ground areas (excluding puddles for now)
        torch::Tensor non_puddle_ground_mask = ground_mask & ~puddle_mask;
        result_image = torch::where(non_puddle_ground_mask, wet_effect, result_image);

        // 4. --- Apply Puddle Reflections ---
        if (puddle_mask.any().item<bool>()) {
            torch::Tensor puddle_base_color = image * darken_multiplier;
            torch::Tensor reflection_color_reshaped = reflection_color_.to(image.options()).view({3, 1, 1});

            // Blend the puddle base color with the reflection color
            torch::Tensor puddle_color = torch::lerp(
                    puddle_base_color,
                    reflection_color_reshaped,
                    reflection_intensity_
            );

            result_image = torch::where(puddle_mask, puddle_color, result_image);
        }

        return torch::clamp(result_image, 0.0, 1.0);
    }

} // namespace xt::transforms::weather