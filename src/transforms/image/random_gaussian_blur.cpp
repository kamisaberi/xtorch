#include <transforms/image/random_gaussian_blur.h>

// --- Example Main (for testing) ---
// #include "transforms/image/random_gaussian_blur.h"
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
//     std::cout << "--- Applying RandomGaussianBlur ---" << std::endl;
//
//     // 2. Define the transform. Kernel 21x21, sigma range [1.0, 5.0].
//     //    We use p=1.0 to guarantee the transform is applied.
//     xt::transforms::image::RandomGaussianBlur blurrer(21, {1.0, 5.0}, 1.0);
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

    RandomGaussianBlur::RandomGaussianBlur() : RandomGaussianBlur(7, {0.1, 2.0}, 0.5) {}

    RandomGaussianBlur::RandomGaussianBlur(
        int kernel_size,
        std::pair<double, double> sigma_range,
        double p)
        : kernel_size_(kernel_size), sigma_range_(sigma_range), p_(p) {

        if (kernel_size_ <= 0 || kernel_size_ % 2 == 0) {
            throw std::invalid_argument("Kernel size must be a positive, odd integer.");
        }
        if (sigma_range_.first < 0 || sigma_range_.second < 0) {
            throw std::invalid_argument("Sigma values must be non-negative.");
        }
        if (p_ < 0.0 || p_ > 1.0) {
            throw std::invalid_argument("Probability p must be between 0.0 and 1.0.");
        }

        std::random_device rd;
        gen_.seed(rd());
    }

    auto RandomGaussianBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // --- Probability Check ---
        std::uniform_real_distribution<> prob_dist(0.0, 1.0);
        if (prob_dist(gen_) >= p_) {
            return tensors.begin()[0]; // Return the original tensor
        }

        // --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("RandomGaussianBlur::forward received an empty list.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to RandomGaussianBlur is not defined.");
        }

        // --- Determine Random Sigma ---
        std::uniform_real_distribution<> sigma_dist(sigma_range_.first, sigma_range_.second);
        double sigma = sigma_dist(gen_);

        // --- Convert to OpenCV Mat (Float for precision) ---
        cv::Mat input_mat_32f = xt::utils::image::tensor_to_mat_float(input_tensor);

        // --- Apply Gaussian Blur ---
        cv::Mat output_mat_32f;
        // For an isotropic (circular) Gaussian, we provide a value for sigmaX and set
        // sigmaY to 0, which tells OpenCV to make it equal to sigmaX.
        cv::GaussianBlur(
            input_mat_32f,
            output_mat_32f,
            cv::Size(kernel_size_, kernel_size_),
            sigma,
            0 // sigmaY = 0 means it will be set equal to sigmaX
        );

        // --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_float(output_mat_32f);

        return output_tensor;
    }

} // namespace xt::transforms::image