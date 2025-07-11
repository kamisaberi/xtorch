#include "include/transforms/image/spatter.h"


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