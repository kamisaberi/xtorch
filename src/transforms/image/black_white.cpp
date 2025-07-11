#include "include/transforms/image/black_white.h"

namespace xt::transforms::image {

    BlackWhite::BlackWhite() : threshold_(0.5f) {}

    BlackWhite::BlackWhite(float threshold) : threshold_(threshold) {
        if (threshold < 0.0f || threshold > 1.0f) {
            // It's good practice to warn the user if the threshold is outside the typical 0-1 range for normalized images.
            // You could also throw an exception if you want to strictly enforce it.
            // For now, we'll just allow it.
        }
    }

    auto BlackWhite::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("BlackWhite::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to BlackWhite is not defined.");
        }
        if (input_tensor.dim() != 3) {
            throw std::invalid_argument("BlackWhite transform expects a 3D image tensor (C, H, W).");
        }

        // --- Create a temporary tensor to work with ---
        torch::Tensor processed_tensor;

        // 2. --- Handle Input Channels ---
        // If the image is 3-channel (color), convert it to grayscale first.
        if (input_tensor.size(0) == 3) {
            // Use the standard luminosity formula: Y = 0.299*R + 0.587*G + 0.114*B
            auto weights = torch::tensor({0.299, 0.587, 0.114}, input_tensor.options())
                               .view({3, 1, 1});
            processed_tensor = (input_tensor * weights).sum(/*dim=*/0, /*keepdim=*/true);
        }
        // If it's already 1-channel (grayscale), we can use it directly.
        else if (input_tensor.size(0) == 1) {
            processed_tensor = input_tensor;
        }
        else {
            throw std::invalid_argument("BlackWhite transform expects a 1-channel (grayscale) or 3-channel (RGB) image.");
        }

        // 3. --- Apply Binary Thresholding ---
        // The most efficient way to threshold in LibTorch is a direct boolean comparison.
        // This creates a boolean tensor where each element is true or false.
        torch::Tensor binary_tensor = (processed_tensor > threshold_);

        // 4. --- Convert Boolean Tensor to Float (0.0 or 1.0) ---
        // Models typically expect float inputs, so we convert the boolean tensor
        // (true/false) to a float tensor (1.0/0.0).
        return binary_tensor.to(processed_tensor.scalar_type());
    }

} // namespace xt::transforms::image