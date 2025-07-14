#include "include/transforms/image/gaussian_blur.h"


// --- Example Main (for testing) ---
// #include "transforms/image/gaussian_blur.h"
// #include <iostream>
// #include <opencv2/highgui.hpp>
// #include "utils/image_conversion.h"
//
// int main() {
//     // 1. Create a sample image with sharp details (a grid).
//     cv::Mat image_mat(256, 256, CV_8UC3, cv::Scalar(255, 255, 255));
//     for (int i = 0; i < image_mat.rows; i += 16) cv::line(image_mat, {0, i}, {image_mat.cols, i}, {0, 0, 0}, 1);
//     for (int i = 0; i < image_mat.cols; i += 16) cv::line(image_mat, {i, 0}, {i, image_mat.rows}, {0, 0, 0}, 1);
//     cv::imwrite("blur_before.png", image_mat);
//     std::cout << "Saved blur_before.png" << std::endl;
//
//     torch::Tensor image = xt::utils::image::mat_to_tensor_float(image_mat);
//
//     std::cout << "--- Applying GaussianBlur ---" << std::endl;
//
//     // 2. Define the transform. Kernel 15x15, sigmaX = 3.0.
//     xt::transforms::image::GaussianBlur blurrer(15, {3.0, 0.0});
//
//     // 3. Apply the transform
//     torch::Tensor blurred_tensor = std::any_cast<torch::Tensor>(blurrer.forward({image}));
//
//     // 4. Save the result.
//     cv::Mat blurred_mat = xt::utils::image::tensor_to_mat_8u(blurred_tensor);
//     cv::imwrite("blur_after.png", blurred_mat);
//     std::cout << "Saved blur_after.png" << std::endl;
//
//     return 0;
// }


namespace xt::transforms::image {

    GaussianBlur::GaussianBlur() : GaussianBlur(0, {0,0}) {}

    GaussianBlur::GaussianBlur(
        int kernel_size,
        std::pair<double, double> sigma)
        : kernel_size_(kernel_size), sigma_(sigma) {

        if (kernel_size_ > 0 && kernel_size_ % 2 == 0) {
            throw std::invalid_argument("Kernel size must be a positive, odd integer, or 0.");
        }
        if (sigma_.first < 0 || sigma_.second < 0) {
            throw std::invalid_argument("Sigma values must be non-negative.");
        }
    }

    auto GaussianBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("GaussianBlur::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to GaussianBlur is not defined.");
        }
        if (kernel_size_ <= 0) {
            // No-op if kernel size is not provided
            return input_tensor;
        }

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);

        // --- Apply Gaussian Blur ---
        cv::Mat output_mat_32f;
        cv::GaussianBlur(
            input_mat_32f,
            output_mat_32f,
            cv::Size(kernel_size_, kernel_size_),
            sigma_.first,  // sigmaX
            sigma_.second  // sigmaY
        );

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat_32f);

        return output_tensor;
    }

} // namespace xt::transforms::image