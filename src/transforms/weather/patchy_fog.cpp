#include "include/transforms/weather/patchy_fog.h"

#include <stdexcept>

// --- Example Main (for testing) ---
// You can uncomment this to create a standalone executable for testing the transform.
/*
#include <iostream>
// #include "xt/utils/image_io.h" // Fictional utility for saving images

int main() {
    // 1. Create a dummy image (e.g., a dark road)
    torch::Tensor image = torch::full({3, 256, 512}, 0.15f);

    std::cout << "--- Applying Procedural Patchy Fog Effect ---" << std::endl;

    // 2. Create the transform. We'll use a low base density but high patch
    //    intensity to make the patches clearly visible.
    xt::transforms::weather::PatchyFog patchy_fog_effect(
        0.1f,                                     // 10% base density
        0.8f,                                     // 80% additional density in patches
        16.0f,                                    // Medium-sized patches
        torch::tensor({0.8, 0.8, 0.8}),           // White fog
        99                                        // Seed
    );

    // 3. Apply the transform.
    torch::Tensor result_image = std::any_cast<torch::Tensor>(patchy_fog_effect.forward({image}));

    std::cout << "Transform applied." << std::endl;
    std::cout << "Result image shape: " << result_image.sizes() << std::endl;
    std::cout << "The resulting image will have areas of light haze mixed with thick fog banks." << std::endl;

    // To see the effect, you would save the output image:
    // xt::utils::save_image(result_image, "patchy_fog_effect.png");

    return 0;
}
*/

namespace xt::transforms::weather {

    PatchyFog::PatchyFog()
            : base_density_(0.2f),
              patch_intensity_(0.6f),
              granularity_(16.0f),
              seed_(0),
              generator_(torch::make_generator<torch::CPUGenerator>(0))
    {
        fog_color_ = torch::tensor({0.8, 0.8, 0.8});
    }

    PatchyFog::PatchyFog(float base_density, float patch_intensity, float granularity, torch::Tensor fog_color, int64_t seed)
            : base_density_(base_density),
              patch_intensity_(patch_intensity),
              granularity_(granularity),
              fog_color_(std::move(fog_color)),
              seed_(seed),
              generator_(torch::make_generator<torch::CPUGenerator>(seed))
    {
        if (base_density_ < 0.0f || base_density_ > 1.0f) {
            throw std::invalid_argument("Base density must be between 0.0 and 1.0.");
        }
        if (patch_intensity_ < 0.0f) {
            throw std::invalid_argument("Patch intensity cannot be negative.");
        }
        if (granularity_ < 1.0f) {
            throw std::invalid_argument("Granularity must be greater than or equal to 1.0.");
        }
        if (fog_color_.numel() != 3) {
            throw std::invalid_argument("Fog color must be a 3-element tensor (R, G, B).");
        }
    }

    auto PatchyFog::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PatchyFog::forward received an empty list of tensors.");
        }
        torch::Tensor image = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!image.defined()) {
            throw std::invalid_argument("Input tensor passed to PatchyFog is not defined.");
        }
        if (image.dim() != 3) {
            throw std::invalid_argument("Input image must be a 3D tensor (C, H, W).");
        }

        // 2. --- Generate Procedural Density Map ---
        auto H = image.size(1);
        auto W = image.size(2);
        auto low_res_H = std::max<int64_t>(1, static_cast<int64_t>(H / granularity_));
        auto low_res_W = std::max<int64_t>(1, static_cast<int64_t>(W / granularity_));

        // Create low-resolution random noise
        torch::Tensor noise = torch::rand({1, 1, low_res_H, low_res_W}, generator_, image.options());

        // Scale it up to full resolution to create a smooth, cloud-like pattern
        torch::Tensor patch_map = torch::upsample_bilinear2d(noise, {H, W}, false).squeeze();

        // 3. --- Calculate Final Density and Blend ---
        // Combine base density with the variable patch map
        torch::Tensor density_map = base_density_ + patch_map * patch_intensity_;
        density_map = torch::clamp(density_map, 0.0, 1.0);

        // Reshape fog color to be broadcastable: {3} -> {3, 1, 1}
        torch::Tensor fog_color_reshaped = fog_color_.to(image.options()).view({3, 1, 1});

        // Blend the image with the fog color using the per-pixel density map
        // final_color = image * (1 - density) + fog_color * density
        torch::Tensor result = image * (1.0f - density_map) + fog_color_reshaped * density_map;

        return torch::clamp(result, 0.0, 1.0);
    }

} // namespace xt::transforms::weather