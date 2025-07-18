#include <transforms/image/noise_injection.h>



// #include "transforms/image/noise_injection.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (a solid gray image)
//     torch::Tensor image = torch::ones({3, 100, 100}) * 0.5;
//
//     // --- Example 1: Add Gaussian Noise ---
//     std::cout << "--- Adding Gaussian Noise ---" << std::endl;
//     xt::transforms::image::NoiseInjection gaussian_noiser("gaussian", {0.0, 0.15});
//
//     torch::Tensor gaussian_noisy_image = std::any_cast<torch::Tensor>(gaussian_noiser.forward({image}));
//
//     std::cout << "Original image mean: " << image.mean().item<float>() << std::endl;
//     std::cout << "Noisy image mean (Gaussian): " << gaussian_noisy_image.mean().item<float>() << " (should be ~0.5)" << std::endl;
//     std::cout << "Noisy image std (Gaussian): " << gaussian_noisy_image.std().item<float>() << " (should be ~0.15)" << std::endl;
//
//
//     // --- Example 2: Add Uniform Noise ---
//     std::cout << "\n--- Adding Uniform Noise ---" << std::endl;
//     // Add noise uniformly from [-0.2, 0.2]
//     xt::transforms::image::NoiseInjection uniform_noiser("uniform", {-0.2, 0.2});
//
//     torch::Tensor uniform_noisy_image = std::any_cast<torch::Tensor>(uniform_noiser.forward({image}));
//
//     std::cout << "Original image mean: " << image.mean().item<float>() << std::endl;
//     std::cout << "Noisy image mean (Uniform): " << uniform_noisy_image.mean().item<float>() << " (should be ~0.5)" << std::endl;
//     // Check the min and max to see the effect of the uniform noise
//     std::cout << "Noisy image min (Uniform): " << uniform_noisy_image.min().item<float>() << " (should be ~0.3)" << std::endl;
//     std::cout << "Noisy image max (Uniform): " << uniform_noisy_image.max().item<float>() << " (should be ~0.7)" << std::endl;
//
//
//     // You could save the images to see the different noise textures.
//     // cv::Mat g_mat = xt::utils::image::tensor_to_mat_8u(gaussian_noisy_image);
//     // cv::imwrite("gaussian_noise_injected.png", g_mat);
//     //
//     // cv::Mat u_mat = xt::utils::image::tensor_to_mat_8u(uniform_noisy_image);
//     // cv::imwrite("uniform_noise_injected.png", u_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    NoiseInjection::NoiseInjection() : noise_type_("gaussian"), params_({0.0, 0.1}) {}

    NoiseInjection::NoiseInjection(const std::string& noise_type, const std::vector<double>& params)
        : noise_type_(noise_type), params_(params) {

        if (noise_type_ == "gaussian") {
            if (params_.size() != 2) {
                throw std::invalid_argument("Gaussian noise requires 2 parameters: {mean, sigma}.");
            }
        } else if (noise_type_ == "uniform") {
            if (params_.size() != 2) {
                throw std::invalid_argument("Uniform noise requires 2 parameters: {low, high}.");
            }
            if (params_[0] >= params_[1]) {
                throw std::invalid_argument("For uniform noise, low must be less than high.");
            }
        } else {
            throw std::invalid_argument("Unsupported noise_type. Must be 'gaussian' or 'uniform'.");
        }
    }

    auto NoiseInjection::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("NoiseInjection::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to NoiseInjection is not defined.");
        }

        // 2. --- Generate Noise Tensor based on Type ---
        torch::Tensor noise;

        if (noise_type_ == "gaussian") {
            double mean = params_[0];
            double sigma = params_[1];
            // randn_like creates noise with mean=0, std=1. We scale and shift it.
            noise = torch::randn_like(input_tensor) * sigma + mean;
        } else if (noise_type_ == "uniform") {
            double low = params_[0];
            double high = params_[1];
            // rand_like creates noise in [0, 1). We scale and shift it to [low, high).
            noise = torch::rand_like(input_tensor) * (high - low) + low;
        }

        // 3. --- Add Noise and Clamp ---
        torch::Tensor noisy_image = input_tensor + noise;
        noisy_image = torch::clamp(noisy_image, 0.0, 1.0);

        return noisy_image;
    }

} // namespace xt::transforms::image