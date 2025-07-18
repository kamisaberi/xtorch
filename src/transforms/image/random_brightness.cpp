#include <transforms/image/random_brightness.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_brightness.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256 * 256).view({1, 256, 256}).repeat({3, 1, 1});
//
//     std::cout << "--- Applying Random Brightness Adjustment ---" << std::endl;
//
//     // 2. Define transform: Brightness factor of 0.8, so the random
//     //    factor will be in the range [max(0, 1-0.8), 1+0.8] = [0.2, 1.8].
//     //    We use p=1.0 to guarantee the transform is applied for the demo.
//     xt::transforms::image::RandomBrightness brightener(0.8, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor brightened_tensor = std::any_cast<torch::Tensor>(brightener.forward({image}));
//
//     // 4. Save the result
//     cv::Mat brightened_mat = xt::utils::image::tensor_to_mat_8u(brightened_tensor);
//     cv::imwrite("brightened_image.png", brightened_mat);
//     std::cout << "Saved brightened_image.png" << std::endl;
//
//     // You could also run it multiple times to see different random results.
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomBrightness::RandomBrightness() : RandomBrightness(0.5, 0.5) {}

    RandomBrightness::RandomBrightness(double brightness_factor, double p)
        : brightness_factor_(brightness_factor), p_(p) {

        if (brightness_factor_ < 0) {
            throw std::invalid_argument("Brightness factor must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomBrightness::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomBrightness::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomBrightness is not defined.");
        }
        if (input_tensor.is_floating_point() == false) {
             throw std::invalid_argument("RandomBrightness requires a floating-point tensor.");
        }

        // --- Determine Random Brightness Factor ---
        double lower_bound = std::max(0.0, 1.0 - brightness_factor_);
        double upper_bound = 1.0 + brightness_factor_;
        std::uniform_real_distribution<> factor_dist(lower_bound, upper_bound);
        double factor = factor_dist(gen_);

        // If factor is 1.0, no change is needed (small optimization).
        if (std::abs(factor - 1.0) < 1e-6) {
            return input_tensor;
        }

        // --- Apply Brightness Adjustment Directly on the Tensor ---
        // This is highly efficient as it avoids any CPU/GPU or format conversions.
        // The operation is equivalent to blending the image with a black image.
        torch::Tensor output_tensor = input_tensor * factor;

        // Clamp the values to ensure they remain in the valid [0, 1] range.
        output_tensor.clamp_(0.0, 1.0);

        return output_tensor;
    }

} // namespace xt::transforms::image