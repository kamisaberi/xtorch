#include <transforms/image/motion_blur.h>

// #include "transforms/image/motion_blur.h"
// #include <iostream>
//
// int main() {
//     // 1. Create a dummy image tensor with a distinct feature
//     torch::Tensor image = torch::zeros({3, 200, 200});
//     // Add a small white square in the center.
//     image.slice(1, 95, 105).slice(2, 95, 105) = 1.0;
//
//     // --- Example 1: Horizontal Motion Blur ---
//     // Angle = 0 degrees, kernel size = 15 pixels
//     xt::transforms::image::MotionBlur horizontal_blurer(15, 0.0);
//     torch::Tensor h_blurred = std::any_cast<torch::Tensor>(horizontal_blurer.forward({image}));
//
//     // You could save the output to see the horizontal blur trail
//     // cv::Mat h_mat = xt::utils::image::tensor_to_mat_8u(h_blurred);
//     // cv::imwrite("horizontal_blur.png", h_mat);
//     std::cout << "Horizontal blur generated." << std::endl;
//
//     // --- Example 2: Diagonal Motion Blur ---
//     // Angle = 45 degrees, kernel size = 21 pixels
//     xt::transforms::image::MotionBlur diagonal_blurer(21, 45.0);
//     torch::Tensor d_blurred = std::any_cast<torch::Tensor>(diagonal_blurer.forward({image}));
//
//     // cv::Mat d_mat = xt::utils::image::tensor_to_mat_8u(d_blurred);
//     // cv::imwrite("diagonal_blur.png", d_mat);
//     std::cout << "Diagonal blur generated." << std::endl;
//
//     // --- Example 3: Random Motion Blur ---
//     // Angle = -1 means a random angle will be picked each time .forward() is called
//     xt::transforms::image::MotionBlur random_blurer(11, -1.0);
//     torch::Tensor r_blurred = std::any_cast<torch::Tensor>(random_blurer.forward({image}));
//
//     // cv::Mat r_mat = xt::utils::image::tensor_to_mat_8u(r_blurred);
//     // cv::imwrite("random_blur.png", r_mat);
//     std::cout << "Random blur generated." << std::endl;
//
//     return 0;
// }



#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xt::transforms::image {

    MotionBlur::MotionBlur() : kernel_size_(7), angle_(-1.0), direction_(1) {}

    MotionBlur::MotionBlur(int kernel_size, double angle, int direction)
        : kernel_size_(kernel_size), angle_(angle), direction_(direction) {

        if (kernel_size_ <= 1 || kernel_size_ % 2 == 0) {
            throw std::invalid_argument("MotionBlur kernel_size must be an odd integer greater than 1.");
        }
    }

    auto MotionBlur::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MotionBlur::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to MotionBlur is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 2. --- Create the Motion Blur Kernel ---
        double current_angle = angle_;
        if (current_angle == -1.0) { // -1 indicates a random angle
            current_angle = cv::theRNG().uniform(0, 360);
        }

        cv::Mat kernel = cv::Mat::zeros(kernel_size_, kernel_size_, CV_32F);

        // Calculate the center of the kernel
        int center = kernel_size_ / 2;

        // Create a line in the kernel matrix
        // The core of this is drawing a line on the kernel and normalizing it.
        // A simple way is to use cv::line
        cv::Point p1, p2;
        double angle_rad = current_angle * M_PI / 180.0;

        double sin_val = sin(angle_rad);
        double cos_val = cos(angle_rad);

        // Calculate endpoints of the line
        p1.x = center - (center * cos_val);
        p1.y = center - (center * sin_val);
        p2.x = center + (center * cos_val);
        p2.y = center + (center * sin_val);

        // Draw the line on the kernel
        cv::line(kernel, p1, p2, cv::Scalar(1.0));

        // Normalize the kernel so that the pixel values' sum is 1
        kernel /= cv::sum(kernel)[0];

        // 3. --- Apply the Filter ---
        cv::Mat blurred_mat;
        cv::filter2D(input_mat, blurred_mat, -1, kernel);

        // 4. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(blurred_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image