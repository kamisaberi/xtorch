#include "include/transforms/image/motion_blur.h"

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