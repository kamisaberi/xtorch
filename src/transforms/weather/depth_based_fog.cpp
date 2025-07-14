#include "include/transforms/weather/depth_based_fog.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>

int main() {
    // 1. Create a dummy image tensor (3 channels, 4x4 pixels).
    // Let's make it a simple green color.
    torch::Tensor image = torch::zeros({3, 4, 4});
    image[1] = 1.0; // Green channel = 1.0

    // 2. Create a dummy depth map tensor (4x4 pixels).
    // Values represent distance. Let's make a gradient from near to far.
    torch::Tensor depth_map = torch::tensor({
        {1, 5, 10, 15},
        {1, 5, 10, 15},
        {20, 25, 30, 40},
        {20, 25, 30, 40}
    }, torch::kFloat32);

    std::cout << "--- Applying Depth-Based Fog ---" << std::endl;
    std::cout << "Original Image (Green Channel):\n" << image[1] << std::endl;
    std::cout << "Depth Map:\n" << depth_map << std::endl;

    // 3. Create and apply the transform.
    // Use the default gray fog (0.5, 0.5, 0.5) with density 0.05.
    xt::transforms::weather::DepthBasedFog fog_applicator;
    torch::Tensor fogged_image = std::any_cast<torch::Tensor>(fog_applicator.forward({image, depth_map}));

    // 4. Print the result.
    // Notice how the green channel values decrease (get foggier) as depth increases.
    std::cout << "\nFogged Image (Green Channel):\n" << fogged_image[1] << std::endl;

    // --- Expected Output (values will be approximate) ---
    // Fogged Image (Green Channel):
    //  0.9512  0.7788  0.6065  0.4724
    //  0.9512  0.7788  0.6065  0.4724
    //  0.3679  0.2865  0.2231  0.1353
    //  0.3679  0.2865  0.2231  0.1353
    // [ CPUFloatType{4,4} ]

    return 0;
}
*/

namespace xt::transforms::weather {

    DepthBasedFog::DepthBasedFog() : density_(0.05f) {
        fog_color_ = torch::tensor({0.5, 0.5, 0.5});
    }

    DepthBasedFog::DepthBasedFog(float fog_density, torch::Tensor fog_color)
            : density_(fog_density), fog_color_(std::move(fog_color)) {

        if (density_ < 0) {
            throw std::invalid_argument("Fog density cannot be negative.");
        }
        if (fog_color_.numel() != 3) {
            throw std::invalid_argument("Fog color must be a 3-element tensor (R, G, B).");
        }
    }

    auto DepthBasedFog::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.size() < 2) {
            throw std::invalid_argument("DepthBasedFog::forward expects two tensors (image and depth map).");
        }

        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);
        torch::Tensor depth = std::any_cast<torch::Tensor>(any_vec[1]);

        if (!image.defined() || !depth.defined()) {
            throw std::invalid_argument("Input tensors passed to DepthBasedFog are not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        // Ensure depth map is broadcastable to image shape
        torch::Tensor depth_map = depth.clone();
        if (depth_map.dim() == 2) {
            depth_map = depth_map.unsqueeze(0); // Add channel dimension: (H, W) -> (1, H, W)
        }
        if (depth_map.size(0) != 1 || depth_map.size(1) != image.size(1) || depth_map.size(2) != image.size(2)) {
            throw std::invalid_argument("Depth map shape must be (H, W) or (1, H, W) matching the image.");
        }

        // 2. --- Calculate Fog Blending Factor ---
        // The fog factor, exp(-density * depth), determines how much of the original image color is kept.
        torch::Tensor fog_factor = torch::exp(-density_ * depth_map);
        // Unsqueeze to make it broadcastable for 3-channel multiplication: (1, H, W) -> (1, 1, H, W)
        // This is not needed if depth_map is already (1,H,W) due to broadcasting rules.

        // 3. --- Apply Fog Formula ---
        // Reshape fog color from {3} to {3, 1, 1} to allow broadcasting over the image's H and W dimensions.
        torch::Tensor fog_color_reshaped = fog_color_.to(image.options()).view({3, 1, 1});

        // final_color = image * fog_factor + fog_color * (1 - fog_factor)
        torch::Tensor fogged_image = image * fog_factor + fog_color_reshaped * (1.0f - fog_factor);

        return fogged_image;
    }

} // namespace xt::transforms::weather