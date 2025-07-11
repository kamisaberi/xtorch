
#include "include/transforms/image/pad.h"

// #include "transforms/image/pad.h" // Your new Pad transform header
// #include "utils/conversion_utils.h" // Your conversion utilities
// #include <iostream>
// #include <torch/torch.h>
//
// // Helper to print tensor shapes for verification
// void print_shape(const std::string& name, const torch::Tensor& t) {
//     std::cout << name << " shape: " << t.sizes() << std::endl;
// }
//
// int main() {
//     // 1. Create a base dummy image tensor to work with
//     torch::Tensor image = torch::ones({3, 100, 100});
//
//     std::cout << "=====================================================" << std::endl;
//     std::cout << "Original Image Shape: " << image.sizes() << std::endl;
//     std::cout << "=====================================================" << std::endl;
//
//     // --- Example 1: Same padding on all sides with a constant value ---
//     std::cout << "\n--- Example 1: Constant Padding (10px on all sides) ---" << std::endl;
//     // Add a 10-pixel black border around the image.
//     // The padding vector has a single element: {10}.
//     xt::transforms::image::Pad padder1({10}, "constant", /*fill_value=*/0.0f);
//
//     torch::Tensor padded1 = std::any_cast<torch::Tensor>(padder1.forward({image}));
//     print_shape("Padded image", padded1);
//     // Expected output shape: [3, 120, 120] (100 + 10_top + 10_bottom, 100 + 10_left + 10_right)
//
//
//     // --- Example 2: Different padding for top/bottom and left/right ---
//     std::cout << "\n--- Example 2: Reflect Padding (30px top/bottom, 20px left/right) ---" << std::endl;
//     // The padding vector has two elements: {top/bottom, left/right}.
//     xt::transforms::image::Pad padder2({30, 20}, "reflect");
//
//     torch::Tensor padded2 = std::any_cast<torch::Tensor>(padder2.forward({image}));
//     print_shape("Padded image", padded2);
//     // New Height = 100 + 30_top + 30_bottom = 160
//     // New Width  = 100 + 20_left + 20_right = 140
//     // Expected output shape: [3, 160, 140]
//
//
//     // --- Example 3: Specific padding for each of the four sides ---
//     std::cout << "\n--- Example 3: Replicate Padding (5px top, 10px bottom, 15px left, 20px right) ---" << std::endl;
//     // The padding vector has four elements: {top, bottom, left, right}.
//     xt::transforms::image::Pad padder3({5, 10, 15, 20}, "replicate");
//
//     torch::Tensor padded3 = std::any_cast<torch::Tensor>(padder3.forward({image}));
//     print_shape("Padded image", padded3);
//     // New Height = 100 + 5_top + 10_bottom = 115
//     // New Width  = 100 + 15_left + 20_right = 135
//     // Expected output shape: [3, 115, 135]
//
//
//     // To visually inspect the results, you would convert these tensors to cv::Mat
//     // and save them. For example, for the "reflect" padding:
//     /*
//     cv::Mat mat_padded2 = xt::utils::image::tensor_to_mat_8u(padded2);
//     // You might need to create a dummy image with some features to see the reflection.
//     // Let's create one.
//     torch::Tensor feature_image = torch::zeros({1, 100, 100});
//     feature_image.slice(1, 10, 40).slice(2, 10, 40) = 1.0; // A square in the top-left
//
//     torch::Tensor padded_feature_image = std::any_cast<torch::Tensor>(padder2.forward({feature_image}));
//
//     cv::Mat mat_original = xt::utils::image::tensor_to_mat_8u(feature_image);
//     cv::Mat mat_padded_reflect = xt::utils::image::tensor_to_mat_8u(padded_feature_image);
//
//     cv::imwrite("original_for_padding.png", mat_original);
//     cv::imwrite("padded_reflect.png", mat_padded_reflect);
//     std::cout << "\nSaved 'padded_reflect.png' to demonstrate reflection padding." << std::endl;
//     */
//
//     return 0;
// }

namespace xt::transforms::image {

    Pad::Pad() : top_(0), bottom_(0), left_(0), right_(0), border_type_flag_(cv::BORDER_CONSTANT), fill_value_(0.0f) {}

    Pad::Pad(const std::vector<int>& padding, const std::string& mode, float fill_value)
        : fill_value_(fill_value) {

        // Parse the padding vector into top, bottom, left, right
        if (padding.size() == 1) {
            top_ = bottom_ = left_ = right_ = padding[0];
        } else if (padding.size() == 2) {
            top_ = bottom_ = padding[0];
            left_ = right_ = padding[1];
        } else if (padding.size() == 4) {
            top_ = padding[0];
            bottom_ = padding[1];
            left_ = padding[2];
            right_ = padding[3];
        } else {
            throw std::invalid_argument("Padding vector must have 1, 2, or 4 elements: {p}, {p_tb, p_lr}, or {top, bottom, left, right}.");
        }

        // Parse the string mode into an OpenCV enum
        if (mode == "constant") {
            border_type_flag_ = cv::BORDER_CONSTANT;
        } else if (mode == "reflect") {
            // Note: OpenCV has two reflect modes. BORDER_REFLECT_101 is usually what people mean.
            border_type_flag_ = cv::BORDER_REFLECT_101;
        } else if (mode == "replicate") {
            border_type_flag_ = cv::BORDER_REPLICATE;
        } else {
            throw std::invalid_argument("Unsupported padding mode. Must be 'constant', 'reflect', or 'replicate'.");
        }
    }

    auto Pad::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation and Conversion to Mat ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("Pad::forward received an empty list of tensors.");
        }
        torch::Tensor input_tensor = std::any_cast<torch::Tensor>(any_vec[0]);

        if (!input_tensor.defined()) {
            throw std::invalid_argument("Input tensor passed to Pad is not defined.");
        }

        cv::Mat input_mat = xt::utils::image::tensor_to_mat_local(input_tensor);

        // 2. --- Apply Padding using cv::copyMakeBorder ---
        cv::Mat padded_mat;

        // The fill value needs to be converted to a cv::Scalar.
        // If the mat has 3 channels, the scalar should have 3 identical values.
        cv::Scalar cv_fill_value;
        if (input_mat.channels() == 3) {
            cv_fill_value = cv::Scalar(fill_value_, fill_value_, fill_value_);
        } else {
            cv_fill_value = cv::Scalar(fill_value_);
        }

        cv::copyMakeBorder(
            input_mat,
            padded_mat,
            top_,
            bottom_,
            left_,
            right_,
            border_type_flag_,
            cv_fill_value // This is only used for BORDER_CONSTANT
        );

        // 3. --- Convert back to LibTorch Tensor ---
        torch::Tensor output_tensor = xt::utils::image::mat_to_tensor_local(padded_mat);

        return output_tensor;
    }

} // namespace xt::transforms::image