#include <transforms/weather/homogeneous_fog.h>

#include <stdexcept>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy image tensor (e.g., a simple landscape)
    torch::Tensor image = torch::zeros({3, 128, 128});
    image[2].slice(0, 64, 128).fill_(0.8); // Blue sky on top
    image[1].slice(0, 0, 64).fill_(0.6);   // Green ground on bottom

    std::cout << "--- Applying Homogeneous Fog ---" << std::endl;
    std::cout << "Original image created (blue sky, green ground)." << std::endl;

    // 2. Create the transform with 50% gray fog.
    xt::transforms::weather::HomogeneousFog fog_applicator(0.5f, torch::tensor({0.5, 0.5, 0.5}));

    // 3. Apply the transform.
    torch::Tensor fogged_image = std::any_cast<torch::Tensor>(fog_applicator.forward({image}));

    // 4. Check the results.
    // The blue channel in the sky was 0.8. After 50% blend with 0.5 gray, it should be:
    // 0.8 * (1 - 0.5) + 0.5 * 0.5 = 0.4 + 0.25 = 0.65
    // The green channel on the ground was 0.6. After blend, it should be:
    // 0.6 * (1 - 0.5) + 0.5 * 0.5 = 0.3 + 0.25 = 0.55
    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << fogged_image.sizes() << std::endl;
    std::cout << "A 'slice' of the blue channel of the fogged image:\n"
              << fogged_image[2].slice(0, 70, 80).slice(1, 0, 10) << std::endl;


    // For a real test, you would save the output image:
    // xt::utils::save_image(fogged_image, "homogeneous_fog_effect.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    HomogeneousFog::HomogeneousFog() : density_(0.4f) {
        fog_color_ = torch::tensor({0.7, 0.7, 0.7});
    }

    HomogeneousFog::HomogeneousFog(float density, torch::Tensor fog_color)
            : density_(density), fog_color_(std::move(fog_color)) {

        if (density_ < 0.0f || density_ > 1.0f) {
            throw std::invalid_argument("Fog density must be between 0.0 and 1.0.");
        }
        if (fog_color_.numel() != 3) {
            throw std::invalid_argument("Fog color must be a 3-element tensor (R, G, B).");
        }
    }

    auto HomogeneousFog::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("HomogeneousFog::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to HomogeneousFog is not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        // If fog is transparent, return the original image
        if (density_ == 0.0f) {
            return image;
        }

        // 2. --- Apply Fog Blending ---
        // Reshape fog color from {3} to {3, 1, 1} to allow broadcasting over the image's H and W dimensions.
        torch::Tensor fog_color_reshaped = fog_color_.to(image.options()).view({3, 1, 1});

        // If fog is fully dense, no need to calculate with original image
        if (density_ == 1.0f) {
            // torch::full_like creates a tensor with the same shape/type as image, filled with the value
            return torch::full_like(image, 0.0) + fog_color_reshaped;
        }

        // The core blending formula: final = image * (1 - density) + fog_color * density
        torch::Tensor result = image * (1.0f - density_) + fog_color_reshaped * density_;

        return result;
    }

} // namespace xt::transforms::weather