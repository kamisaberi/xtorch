#include <transforms/image/brightness.h>


// #include "transforms/image/brightness.h"
// #include <iostream>
//
// // Helper function to generate a random float in a range
// float random_float(float min, float max) {
//     return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
// }
//
// int main() {
//     // Seed for reproducibility
//     srand(static_cast<unsigned>(time(0)));
//
//     // 1. Create a dummy image tensor (e.g., a gray image)
//     torch::Tensor image = torch::ones({3, 100, 100}) * 0.5;
//
//     // --- Example 1: Increase Brightness ---
//     xt::transforms::image::Brightness brighter(0.3f); // Add 0.3
//     torch::Tensor bright_image = std::any_cast<torch::Tensor>(brighter.forward({image}));
//     // Expected mean pixel value: 0.5 + 0.3 = 0.8
//     std::cout << "Mean of brightened image: " << bright_image.mean().item<float>() << std::endl;
//
//     // --- Example 2: Decrease Brightness ---
//     xt::transforms::image::Brightness darker(-0.4f); // Subtract 0.4
//     torch::Tensor dark_image = std::any_cast<torch::Tensor>(darker.forward({image}));
//     // Expected mean pixel value: 0.5 - 0.4 = 0.1
//     std::cout << "Mean of darkened image: " << dark_image.mean().item<float>() << std::endl;
//
//     // --- Example 3: Random Brightness for Data Augmentation ---
//     // Generate a random delta, e.g., between -0.2 and 0.2
//     float random_delta = random_float(-0.2f, 0.2f);
//     std::cout << "Applying random delta of: " << random_delta << std::endl;
//
//     xt::transforms::image::Brightness random_adjuster(random_delta);
//     torch::Tensor random_image = std::any_cast<torch::Tensor>(random_adjuster.forward({image}));
//     std::cout << "Mean of randomly adjusted image: " << random_image.mean().item<float>() << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    // Default constructor: delta = 0 means no change.
    Brightness::Brightness() : delta_(0.0f) {}

    Brightness::Brightness(float delta) : delta_(delta) {}

    auto Brightness::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Brightness::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Brightness is not defined.");
        }

        // 2. --- Add Delta to Adjust Brightness ---
        // This is a simple, element-wise addition.
        torch::Tensor brightened_tensor = input_tensor + delta_;

        // 3. --- Clamp to Valid Range ---
        // It's crucial to ensure pixel values stay within the valid range.
        // For normalized float images, this is typically [0, 1].
        brightened_tensor = torch::clamp(brightened_tensor, 0.0, 1.0);

        return brightened_tensor;
    }

} // namespace xt::transforms::image