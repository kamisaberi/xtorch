#include "include/transforms/image/random_contrast.h"


// --- Example Main (for testing) ---
// #include "transforms/image/random_contrast.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a gradient image to see the effect clearly.
//     torch::Tensor image = torch::linspace(0, 1, 256 * 256).view({1, 256, 256}).repeat({3, 1, 1});
//
//     std::cout << "--- Applying Random Contrast Adjustment ---" << std::endl;
//
//     // 2. Define transform: Contrast limit of 0.5, so the random
//     //    factor will be in the range [max(0, 1-0.5), 1+0.5] = [0.5, 1.5].
//     //    We use p=1.0 to guarantee the transform is applied for the demo.
//     xt::transforms::image::RandomContrast contraster(0.5, 1.0);
//
//     // 3. Apply the transform
//     torch::Tensor contrasted_tensor = std::any_cast<torch::Tensor>(contraster.forward({image}));
//
//     // 4. Save the result
//     cv::Mat contrasted_mat = xt::utils::image::tensor_to_mat_8u(contrasted_tensor);
//     cv::imwrite("contrasted_image.png", contrasted_mat);
//     std::cout << "Saved contrasted_image.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomContrast::RandomContrast() : RandomContrast(0.2, 0.5) {}

    RandomContrast::RandomContrast(double contrast_limit, double p)
        : contrast_limit_(contrast_limit), p_(p) {

        if (contrast_limit_ < 0) {
            throw std::invalid_argument("Contrast limit must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomContrast::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomContrast::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomContrast is not defined.");
        }
        if (input_tensor.is_floating_point() == false) {
             throw std::invalid_argument("RandomContrast requires a floating-point tensor.");
        }

        // --- Determine Random Contrast Factor ---
        double lower_bound = std::max(0.0, 1.0 - contrast_limit_);
        double upper_bound = 1.0 + contrast_limit_;
        std::uniform_real_distribution<> factor_dist(lower_bound, upper_bound);
        double factor = factor_dist(gen_);

        // If factor is 1.0, no change is needed (small optimization).
        if (std::abs(factor - 1.0) < 1e-6) {
            return input_tensor;
        }

        // --- Apply Contrast Adjustment Directly on the Tensor ---
        // 1. Calculate the mean intensity of the image (the "gray" value).
        //    We compute the mean across the color channels (dim 0).
        torch::Tensor grayscale = torch::mean(input_tensor, /*dim=*/0, /*keepdim=*/true);

        // 2. Blend the original image with the mean grayscale image.
        //    This is a linear interpolation: lerp(start, end, weight)
        //    output = grayscale * (1 - factor) + input_tensor * factor
        torch::Tensor output_tensor = torch::lerp(grayscale, input_tensor, factor);

        // 3. Clamp the values to ensure they remain in the valid [0, 1] range.
        output_tensor.clamp_(0.0, 1.0);

        return output_tensor;
    }

} // namespace xt::transforms::image