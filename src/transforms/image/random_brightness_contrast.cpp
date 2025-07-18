#include <transforms/image/random_brightness_contrast.h>


// --- Example Main (for testing) ---
// #include "transforms/image/random_brightness_contrast.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256 * 256).view({1, 256, 256}).repeat({3, 1, 1});
//
//     std::cout << "--- Applying Random Brightness & Contrast Adjustment ---" << std::endl;
//
//     // 2. Define transform: Brightness +/- 0.3, Contrast +/- 0.4.
//     //    We use p=1.0 to guarantee the transform is applied for the demo.
//     xt::transforms::image::RandomBrightnessContrast transformer(0.3, 0.4, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor transformed_tensor = std::any_cast<torch::Tensor>(transformer.forward({image}));
//
//     // 4. Save the result
//     cv::Mat transformed_mat = xt::utils::image::tensor_to_mat_8u(transformed_tensor);
//     cv::imwrite("brightness_contrast_image.png", transformed_mat);
//     std::cout << "Saved brightness_contrast_image.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomBrightnessContrast::RandomBrightnessContrast()
        : RandomBrightnessContrast(0.2, 0.2, 0.5) {}

    RandomBrightnessContrast::RandomBrightnessContrast(
        double brightness_limit,
        double contrast_limit,
        double p)
        : brightness_limit_(brightness_limit), contrast_limit_(contrast_limit), p_(p) {

        if (brightness_limit_ < 0) {
            throw std::invalid_argument("Brightness limit must be non-negative.");
        }
        if (contrast_limit_ < 0) {
            throw std::invalid_argument("Contrast limit must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomBrightnessContrast::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomBrightnessContrast::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomBrightnessContrast is not defined.");
        }
        if (input_tensor.is_floating_point() == false) {
             throw std::invalid_argument("RandomBrightnessContrast requires a floating-point tensor.");
        }

        // --- Determine Random Factors ---
        // Brightness factor is an additive term, chosen from [-limit, limit]
        std::uniform_real_distribution<> brightness_dist(-brightness_limit_, brightness_limit_);
        double brightness_factor = brightness_dist(gen_);

        // Contrast factor is a multiplicative term, chosen from [1-limit, 1+limit]
        double lower_contrast = 1.0 - contrast_limit_;
        double upper_contrast = 1.0 + contrast_limit_;
        std::uniform_real_distribution<> contrast_dist(lower_contrast, upper_contrast);
        double contrast_factor = contrast_dist(gen_);

        // --- Apply Transformation Directly on the Tensor ---
        // Formula: output = image * contrast_factor + brightness_factor
        torch::Tensor output_tensor = input_tensor * contrast_factor + brightness_factor;

        // Clamp the values to ensure they remain in the valid [0, 1] range.
        output_tensor.clamp_(0.0, 1.0);

        return output_tensor;
    }

} // namespace xt::transforms::image