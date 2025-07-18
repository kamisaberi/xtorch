#include <transforms/image/random_adjust_sharpness.h>



// --- Example Main (for testing) ---
// #include "transforms/image/random_adjust_sharpness.h"
// #include <iostream>
// #include <opencv2/highgui.hpp> // For cv::imread and cv::imwrite
// #include "utils/image_conversion.h" // For the main function's use
//
// int main() {
//     // 1. Load a sample image.
//     // Make sure you have an image file named "sample_image.png" in your execution directory.
//     cv::Mat image_mat = cv::imread("sample_image.png", cv::IMREAD_COLOR);
//     if (image_mat.empty()) {
//         std::cerr << "Error: Could not load sample_image.png" << std::endl;
//         // As a fallback, create a gradient image.
//         torch::Tensor image_tensor = torch::linspace(0, 1, 512).view({1, -1}).repeat({3, 256, 1});
//         image_mat = xt::utils::image::tensor_to_mat_8u(image_tensor);
//         cv::cvtColor(image_mat, image_mat, cv::COLOR_RGB2BGR);
//     }
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     // --- Example 1: Sharpening ---
//     // We use p=1.0 to guarantee the transform is applied.
//     // We use a factor of 2.0, so the adjustment will be in [max(0, 1-2), 1+2] = [0, 3].
//     // This allows for strong sharpening.
//     std::cout << "--- Applying Random Sharpness Adjustment (Factor 2.0) ---" << std::endl;
//     xt::transforms::image::RandomAdjustSharpness sharpener(2.0, 1.0);
//
//     torch::Tensor sharpened_tensor = std::any_cast<torch::Tensor>(sharpener.forward({image}));
//
//     cv::Mat sharpened_mat = xt::utils::image::tensor_to_mat_8u(sharpened_tensor);
//     cv::imwrite("sharpened_image.png", sharpened_mat);
//     std::cout << "Saved sharpened_image.png" << std::endl;
//
//     // --- Example 2: Blurring ---
//     // We use a factor of 0.8, so the adjustment will be in [max(0, 1-0.8), 1+0.8] = [0.2, 1.8].
//     // It can still sharpen, but is more likely to blur.
//     std::cout << "\n--- Applying Random Sharpness Adjustment (Factor 0.8) ---" << std::endl;
//     xt::transforms::image::RandomAdjustSharpness blurrer(0.8, 1.0);
//
//     torch::Tensor blurred_tensor = std::any_cast<torch::Tensor>(blurrer.forward({image}));
//
//     cv::Mat blurred_mat = xt::utils::image::tensor_to_mat_8u(blurred_tensor);
//     cv::imwrite("blurred_image.png", blurred_mat);
//     std::cout << "Saved blurred_image.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    RandomAdjustSharpness::RandomAdjustSharpness() : sharpness_factor_(1.0), p_(0.5) {
        // Seed the random number generator
        std::random_device rd;
        gen_.seed(rd());
    }

    RandomAdjustSharpness::RandomAdjustSharpness(double sharpness_factor, double p)
        : sharpness_factor_(sharpness_factor), p_(p) {
        if (sharpness_factor_ < 0) {
            throw std::invalid_argument("Sharpness factor must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }
        // Seed the random number generator
        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomAdjustSharpness::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomAdjustSharpness::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomAdjustSharpness is not defined.");
        }

        // --- Determine Random Sharpness Factor ---
        double lower_bound = std::max(0.0, 1.0 - sharpness_factor_);
        double upper_bound = 1.0 + sharpness_factor_;
        std::uniform_real_distribution<> factor_dist(lower_bound, upper_bound);
        double factor = factor_dist(gen_);

        // If factor is 1.0, no change is needed.
        if (std::abs(factor - 1.0) < 1e-6) {
            return input_tensor;
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        // Blending operations work best on floating point data to avoid clamping issues.
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);

        // --- Create Blurred "Degenerate" Image ---
        cv::Mat blurred_mat;
        // A 3x3 Gaussian kernel is a standard choice for a basic blur.
        cv::GaussianBlur(input_mat_32f, blurred_mat, cv::Size(0, 0), 3);

        // --- Blend Original and Blurred Images ---
        // The formula is: output = original * factor + blurred * (1 - factor)
        // cv::addWeighted implements this efficiently: dst = src1*alpha + src2*beta + gamma
        cv::Mat output_mat;
        cv::addWeighted(input_mat_32f, factor, blurred_mat, 1.0 - factor, 0.0, output_mat);

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image