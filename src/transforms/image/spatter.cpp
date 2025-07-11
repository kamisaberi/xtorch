#include "include/transforms/image/spatter.h"

// #include "transforms/image/spatter.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a gradient to see the effect clearly
//     torch::Tensor image = torch::linspace(0, 1, 200).view({1, -1}).repeat({3, 200, 1});
//
//     std::cout << "Original image std dev: " << image.std().item<float>() << std::endl;
//
//     // 2. Instantiate the transform to add spatter/speckle noise.
//     // A sigma of 0.2 will create a noticeable effect.
//     xt::transforms::image::Spatter spatterer(0.0, 0.2);
//
//     // 3. Apply the transform
//     std::any result_any = spatterer.forward({image});
//     torch::Tensor spattered_image = std::any_cast<torch::Tensor>(result_any);
//
//     // 4. Check the output
//     std::cout << "Spattered image shape: " << spattered_image.sizes() << std::endl;
//     std::cout << "Spattered image std dev: " << spattered_image.std().item<float>() << std::endl;
//     // The standard deviation should be higher than the original.
//
//     // You can save the output to see the effect. The noise will be more intense
//     // in the brighter parts of the gradient (right side) and less intense in the
//     // darker parts (left side).
//     // cv::Mat output_mat = xt::utils::image::tensor_to_mat_8u(spattered_image);
//     // cv::imwrite("spatter_image.png", output_mat);
//
//     return 0;
// }

namespace xt::transforms::image {

    Spatter::Spatter() : mean_(0.0), sigma_(0.1) {}

    Spatter::Spatter(double mean, double sigma) : mean_(mean), sigma_(sigma) {}

    auto Spatter::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Spatter::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Spatter is not defined.");
        }

        // If sigma is 0, there is no noise, so we can return early.
        if (sigma_ == 0.0) {
            return input_tensor;
        }

        // 2. --- Generate Multiplicative Noise ---
        // Create a tensor of random numbers from a standard normal distribution (mean=0, std=1).
        torch::Tensor noise = torch::randn_like(input_tensor);

        // Scale and shift the noise to the desired mean and sigma.
        noise = noise * sigma_ + mean_;

        // For multiplicative noise, the noise field is centered around 1.0.
        noise += 1.0;

        // 3. --- Multiply Noise with Image and Clamp ---
        torch::Tensor spattered_image = input_tensor * noise;

        // Clamp the result to the valid range for image data (e.g., [0, 1] for normalized floats).
        spattered_image = torch::clamp(spattered_image, 0.0, 1.0);

        return spattered_image;
    }

} // namespace xt::transforms::image