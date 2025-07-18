#include <transforms/weather/dust_sand_clouds.h>
#include <stdexcept>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>
// You would need an image loading utility for a real test,
// but we can fake an image with a simple tensor.
// #include "xt/utils/image_io.h"

int main() {
    // 1. Create a dummy image tensor (e.g., a blue sky)
    torch::Tensor image = torch::zeros({3, 128, 128});
    image[2] = 0.8; // Blue channel
    image[1] = 0.6; // Green channel

    std::cout << "--- Applying Procedural Dust Cloud Effect ---" << std::endl;

    // 2. Create the transform with a sandy color and moderate density.
    // RGB for tan/sandy color is approx (210, 180, 140)
    torch::Tensor sand_color = torch::tensor({210.0f/255.0f, 180.0f/255.0f, 140.0f/255.0f});
    xt::transforms::weather::DustSandClouds sand_storm(0.75f, 16.0f, sand_color, 42);

    // 3. Apply the transform
    torch::Tensor result_image = std::any_cast<torch::Tensor>(sand_storm.forward({image}));

    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << result_image.sizes() << std::endl;
    std::cout << "The resulting image is a blend of the original 'sky' and the generated sand storm." << std::endl;

    // For a real test, you would save the output image:
    // xt::utils::save_image(result_image, "sand_storm_effect.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    DustSandClouds::DustSandClouds()
            : density_(0.7f),
              granularity_(16.0f),
              seed_(0) {
        // Default sandy color (R=210, G=180, B=140)
        dust_color_ = torch::tensor({210.0f / 255.0f, 180.0f / 255.0f, 140.0f / 255.0f});
    }

    DustSandClouds::DustSandClouds(float density, float granularity, torch::Tensor dust_color, int64_t seed)
            : density_(density),
              granularity_(granularity),
              dust_color_(std::move(dust_color)),
              seed_(seed) {

        if (density_ < 0.0f || density_ > 1.0f) {
            throw std::invalid_argument("Density must be between 0.0 and 1.0.");
        }
        if (granularity_ < 1.0f) {
            throw std::invalid_argument("Granularity must be greater than or equal to 1.0.");
        }
        if (dust_color_.numel() != 3) {
            throw std::invalid_argument("Dust color must be a 3-element tensor (R, G, B).");
        }
    }

    auto DustSandClouds::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("DustSandClouds::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to DustSandClouds is not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        // 2. --- Procedural Noise Generation ---
        auto H = image.size(1);
        auto W = image.size(2);
        auto low_res_H = static_cast<int64_t>(H / granularity_);
        auto low_res_W = static_cast<int64_t>(W / granularity_);

        // Ensure low-res dimensions are at least 1
        low_res_H = std::max<int64_t>(1, low_res_H);
        low_res_W = std::max<int64_t>(1, low_res_W);

        // --- Start of fix ---
        // Create a default generator instance. This is more robust across LibTorch versions.
        torch::Generator generator;
        // Set the seed for reproducible results.
        generator.set_current_seed(seed_);
        // --- End of fix ---

        // Create a small tensor of random values
        torch::Tensor noise = torch::rand({1, 1, low_res_H, low_res_W}, generator, image.options());

        // Scale it up to the full image size using bilinear interpolation to create smooth clouds
        torch::Tensor cloud_mask = torch::upsample_bilinear2d(
                noise,
                {H, W},
                false // align_corners
        );
        // Squeeze to remove batch and channel dims for broadcasting: (1, 1, H, W) -> (H, W)
        cloud_mask = cloud_mask.squeeze();

        // 3. --- Blending ---
        // Calculate the opacity of the dust at each pixel
        torch::Tensor alpha = cloud_mask * density_;

        // Reshape color to be broadcastable: {3} -> {3, 1, 1}
        torch::Tensor dust_color_reshaped = dust_color_.to(image.options()).view({3, 1, 1});

        // Blend the image with the dust color using the alpha mask
        // final_color = original_image * (1 - alpha) + dust_color * alpha
        torch::Tensor result = image * (1.0f - alpha) + dust_color_reshaped * alpha;

        // Ensure the output values are clamped to the valid [0, 1] range
        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather