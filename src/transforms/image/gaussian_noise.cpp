#include <transforms/image/gaussian_noise.h>



// #include "transforms/image/gaussian_noise.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor (e.g., a solid gray image)
//     torch::Tensor image = torch::ones({3, 100, 100}) * 0.5;
//
//     std::cout << "Original image standard deviation: " << image.std().item<float>() << std::endl;
//     // Expected output: 0.0 (because it's a solid color)
//
//     // 2. Instantiate the transform to add noise with a standard deviation of 0.15
//     xt::transforms::image::GaussianNoise noiser(0.0, 0.15);
//
//     // 3. Apply the transform
//     std::any result_any = noiser.forward({image});
//     torch::Tensor noisy_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Noisy image shape: " << noisy_image.sizes() << std::endl;
//     std::cout << "Noisy image standard deviation: " << noisy_image.std().item<float>() << std::endl;
//     // Expected output: A value close to 0.15
//
//     // You could save the output image to visually inspect the noise effect.
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(noisy_image);
//     // cv::imwrite("noisy_image.png", output_mat);
//
//     return 0;
// }


namespace xt::transforms::image {

    GaussianNoise::GaussianNoise() : mean_(0.0), sigma_(0.1) {}

    GaussianNoise::GaussianNoise(double mean, double sigma) : mean_(mean), sigma_(sigma) {}

    auto GaussianNoise::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GaussianNoise::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to GaussianNoise is not defined.");
        }

        // If sigma is 0, there is no noise to add, so we can return early.
        if (sigma_ == 0.0) {
            return input_tensor;
        }

        // 2. --- Generate Gaussian Noise ---
        // Create a tensor of random numbers from a standard normal distribution (mean=0, std=1)
        // with the same size and device as the input tensor.
        torch::Tensor noise = torch::randn_like(input_tensor);

        // 3. --- Scale and Shift Noise ---
        // Adjust the noise to match the desired mean and sigma.
        noise = noise * sigma_ + mean_;

        // 4. --- Add Noise to Image and Clamp ---
        torch::Tensor noisy_image = input_tensor + noise;

        // Clamp the result to the valid range for image data (e.g., [0, 1] for normalized floats).
        noisy_image = torch::clamp(noisy_image, 0.0, 1.0);

        return noisy_image;
    }

} // namespace xt::transforms::image