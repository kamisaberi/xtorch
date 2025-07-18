#include <transforms/image/pad_if_needed.h>
//
// #include "transforms/image/pad_if_needed.h"
// #include <iostream>
//
// void process_image(const torch::Tensor& image, xt::transforms::image::PadIfNeeded& padder) {
//     std::cout << "Original shape: " << image.sizes();
//     torch::Tensor result = std::any_cast<torch::Tensor>(padder.forward({image}));
//     std::cout << " -> Padded shape: " << result.sizes() << std::endl;
// }
//
// int main() {
//     // 1. Define the minimum required size for our pipeline
//     int min_height = 256;
//     int min_width = 256;
//
//     // 2. Instantiate the transform
//     xt::transforms::image::PadIfNeeded padder(min_height, min_width, "reflect");
//
//     // --- Example 1: An image that is too small in both dimensions ---
//     std::cout << "--- Case 1: Small Image ---" << std::endl;
//     torch::Tensor small_image = torch::rand({3, 100, 150});
//     process_image(small_image, padder);
//     // Expected output shape: [3, 256, 256]
//
//     // --- Example 2: An image that is too small in one dimension ---
//     std::cout << "\n--- Case 2: Tall, Thin Image ---" << std::endl;
//     torch::Tensor tall_image = torch::rand({3, 300, 100});
//     process_image(tall_image, padder);
//     // Height (300) is fine. Width (100) needs padding.
//     // Expected output shape: [3, 300, 256]
//
//     // --- Example 3: An image that is large enough ---
//     std::cout << "\n--- Case 3: Large Image ---" << std::endl;
//     torch::Tensor large_image = torch::rand({3, 512, 512});
//     process_image(large_image, padder);
//     // No padding needed.
//     // Expected output shape: [3, 512, 512]
//
//     return 0;
// }

namespace xt::transforms::image {

    PadIfNeeded::PadIfNeeded()
        : min_height_(256), min_width_(256), border_type_flag_(cv::BORDER_CONSTANT), fill_value_(0.0f) {}

    PadIfNeeded::PadIfNeeded(int min_height, int min_width, const std::string& border_mode, float fill_value)
        : min_height_(min_height), min_width_(min_width), fill_value_(fill_value) {

        if (min_height_ <= 0 || min_width_ <= 0) {
            throw std::invalid_argument("min_height and min_width must be positive.");
        }

        if (border_mode == "constant") {
            border_type_flag_ = cv::BORDER_CONSTANT;
        } else if (border_mode == "reflect") {
            border_type_flag_ = cv::BORDER_REFLECT_101;
        } else if (border_mode == "replicate") {
            border_type_flag_ = cv::BORDER_REPLICATE;
        } else {
            throw std::invalid_argument("Unsupported padding mode. Must be 'constant', 'reflect', or 'replicate'.");
        }
    }

    auto PadIfNeeded::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("PadIfNeeded::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined() || input_tensor.dim() != 3) {
            throw std::invalid_argument("PadIfNeeded expects a defined 3D image tensor (C, H, W).");
        }

        const int64_t H = input_tensor.size(1);
        const int64_t W = input_tensor.size(2);

        // 2. --- Calculate Required Padding ---
        int64_t pad_h = std::max((int64_t)0, min_height_ - H);
        int64_t pad_w = std::max((int64_t)0, min_width_ - W);

        // If no padding is needed, return the original image
        if (pad_h == 0 && pad_w == 0) {
            return input_tensor;
        }

        // 3. --- Apply Symmetrical Padding ---
        // We add half the padding to the top/left and the other half to the bottom/right.
        // Integer division handles the distribution.
        int top = pad_h / 2;
        int bottom = pad_h - top;
        int left = pad_w / 2;
        int right = pad_w - left;

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);
        cv::Mat padded_mat;

        cv::Scalar cv_fill_value;
        if (input_mat.channels() == 3) {
            cv_fill_value = cv::Scalar(fill_value_, fill_value_, fill_value_);
        } else {
            cv_fill_value = cv::Scalar(fill_value_);
        }

        cv::copyMakeBorder(
            input_mat, padded_mat,
            top, bottom, left, right,
            border_type_flag_, cv_fill_value
        );

        // 4. --- Convert back to LibTorch Tensor ---
        return xt::utils::image::mat_to_tensor_local(padded_mat);
    }

} // namespace xt::transforms::image