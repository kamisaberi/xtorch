#include "include/transforms/image/invert.h"

// --- Example Main (for testing) ---
// #include "transforms/image/invert.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 256, 1});
//     cv::imwrite("invert_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved invert_before.png" << std::endl;
//
//     std::cout << "--- Applying Invert ---" << std::endl;
//
//     // 2. Define the transform. It has no parameters.
//     xt::transforms::image::Invert inverter;
//
//     // 3. Apply the transform
//     torch::Tensor inverted_tensor = std::any_cast<torch::Tensor>(inverter.forward({image}));
//
//     // 4. Save the result. The gradient should be reversed.
//     cv::Mat inverted_mat = xt::utils::image::tensor_to_mat_8u(inverted_tensor);
//     cv::imwrite("invert_after.png", inverted_mat);
//     std::cout << "Saved invert_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    Invert::Invert() {
        // No parameters to initialize.
    }

    auto Invert::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Invert::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to Invert is not defined.");
        }
        if (img.is_floating_point() == false) {
             throw std::invalid_argument("Invert requires a floating-point tensor in the [0, 1] range.");
        }

        // --- Apply Inversion ---
        // For a float tensor in [0, 1], inversion is simply 1.0 - tensor.
        return 1.0 - img;
    }

} // namespace xt::transforms::image