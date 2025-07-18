#include <transforms/image/random_invert.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_invert.h"
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
//     std::cout << "--- Applying RandomInvert ---" << std::endl;
//
//     // 2. Define the transform, with p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomInvert inverter(1.0);
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

    RandomInvert::RandomInvert() : RandomInvert(0.5) {}

    RandomInvert::RandomInvert(double p) : p_(p) {
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomInvert::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomInvert::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomInvert is not defined.");
        }
        if (img.is_floating_point() == false) {
             throw std::invalid_argument("RandomInvert requires a floating-point tensor in the [0, 1] range.");
        }

        // --- Apply Inversion ---
        // For a float tensor in [0, 1], inversion is simply 1.0 - tensor.
        return 1.0 - img;
    }

} // namespace xt::transforms::image