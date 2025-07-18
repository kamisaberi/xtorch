#include <transforms/image/random_gamma.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_gamma.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the non-linear effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256).view({1, -1}).repeat({3, 256, 1});
//     cv::imwrite("gamma_before.png", xt::utils::image::tensor_to_mat_8u(image));
//     std::cout << "Saved gamma_before.png" << std::endl;
//
//     // --- Example 1: Make image darker (gamma > 1.0) ---
//     std::cout << "\n--- Applying Gamma Correction (Darken) ---" << std::endl;
//     // Range [1.5, 2.5] will always result in a darker image.
//     xt::transforms::image::RandomGamma darker({1.5, 2.5}, 1.0);
//     torch::Tensor darker_tensor = std::any_cast<torch::Tensor>(darker.forward({image}));
//     cv::imwrite("gamma_darker.png", xt::utils::image::tensor_to_mat_8u(darker_tensor));
//     std::cout << "Saved gamma_darker.png" << std::endl;
//
//     // --- Example 2: Make image brighter (gamma < 1.0) ---
//     std::cout << "\n--- Applying Gamma Correction (Brighten) ---" << std::endl;
//     // Range [0.4, 0.7] will always result in a brighter image.
//     xt::transforms::image::RandomGamma brighter({0.4, 0.7}, 1.0);
//     torch::Tensor brighter_tensor = std::any_cast<torch::Tensor>(brighter.forward({image}));
//     cv::imwrite("gamma_brighter.png", xt::utils::image::tensor_to_mat_8u(brighter_tensor));
//     std::cout << "Saved gamma_brighter.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomGamma::RandomGamma() : RandomGamma({0.8, 1.2}, 0.5) {}

    RandomGamma::RandomGamma(std::pair<double, double> gamma_range, double p)
        : gamma_range_(gamma_range), p_(p) {

        if (gamma_range_.first <= 0 || gamma_range_.second <= 0) {
            throw std::invalid_argument("Gamma values must be positive.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomGamma::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomGamma::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomGamma is not defined.");
        }
        if (input_tensor.is_floating_point() == false) {
             throw std::invalid_argument("RandomGamma requires a floating-point tensor.");
        }

        // --- Determine Random Gamma Value ---
        std::uniform_real_distribution<> gamma_dist(gamma_range_.first, gamma_range_.second);
        double gamma = gamma_dist(gen_);

        // If gamma is 1.0, no change is needed (small optimization).
        if (std::abs(gamma - 1.0) < 1e-6) {
            return input_tensor;
        }

        // --- Apply Gamma Correction Directly on the Tensor ---
        // The formula output = input ^ gamma is applied element-wise.
        return torch::pow(input_tensor, gamma);
    }

} // namespace xt::transforms::image