#include <transforms/image/random_solarize.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_solarize.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 256, 1});
//     cv::imwrite("solarize_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved solarize_before.png" << std::endl;
//
//     std::cout << "--- Applying RandomSolarize ---" << std::endl;
//
//     // 2. Define the transform. Invert all pixels with a value > 0.6.
//     //    Use p=1.0 to guarantee it runs.
//     xt::transforms::image::RandomSolarize solarizer(0.6, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor solarized_tensor = std::any_cast<torch::Tensor>(solarizer.forward({image}));
//
//     // 4. Save the result. The right side of the gradient should be inverted.
//     cv::Mat solarized_mat = xt::utils::image::tensor_to_mat_8u(solarized_tensor);
//     cv::imwrite("solarize_after.png", solarized_mat);
//     std::cout << "Saved solarize_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomSolarize::RandomSolarize() : RandomSolarize(0.5, 0.5) {}

    RandomSolarize::RandomSolarize(double threshold, double p)
        : threshold_(threshold), p_(p) {

        if (threshold_ < 0.0 || threshold_ > 1.0) {
            throw std::invalid_argument("Threshold must be between 0.0 and 1.0.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomSolarize::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomSolarize::forward received an empty list.");
        }
        torch::Tensor img = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!img.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomSolarize is not defined.");
        }
        if (img.is_floating_point() == false) {
             throw std::invalid_argument("RandomSolarize requires a floating-point tensor in the [0, 1] range.");
        }

        // --- Apply Solarization ---
        // Use torch::where to select between original image and its inverse based on the threshold.
        // where(condition, value_if_true, value_if_false)
        // We want to keep the original value if input < threshold.
        return torch::where(img < threshold_, img, 1.0 - img);
    }

} // namespace xt::transforms::image